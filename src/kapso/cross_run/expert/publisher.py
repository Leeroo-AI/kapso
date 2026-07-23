"""Domain authority for crash-safe expert release publication."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from kapso.cross_run.canonical import (
    normalize_utc_timestamp,
    require_content_id,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    ExpertBaseReleaseManifest,
    ExpertPromotionState,
    PublicationArtifactKind,
    ScopeRepositorySettings,
)
from kapso.cross_run.expert.providers import GitHubExpertCurrentReleaseProvider
from kapso.cross_run.expert.release import (
    EXPERT_RELEASE_CONTROL_ARCHIVE,
    EXPERT_RELEASE_EVIDENCE_ARCHIVE,
    EXPERT_RELEASE_MANIFEST_PATH,
    EXPERT_RELEASE_SOURCE_ARCHIVE,
    ExpertReleaseAssembler,
    ExpertReleasePackage,
)
from kapso.cross_run.expert.release_contracts import (
    ExpertReleasePublicationPlan,
    ExpertReleasePublicationStaleResolution,
)
from kapso.cross_run.expert.promotion_authority import (
    ExpertPublicationEligibilityCoordinator,
    ExpertPublicationSecurityDenylistAuthority,
)
from kapso.cross_run.expert.promotion_authority_contracts import (
    ExpertPublicationEligibilityAuthorityFence,
    ExpertPublicationEligibilityStageResultRecord,
)
from kapso.cross_run.expert.task_evaluation_authority_contracts import (
    TaskEvaluationCurrentReleaseObservation,
)
from kapso.cross_run.github.materializer import GitHubArtifactMaterializer
from kapso.cross_run.github.publisher import (
    AutonomousGitHubPublisher,
    PublicationEnvelope,
    PublicationTelemetry,
    ReleaseAssetInput,
)
from kapso.cross_run.github.resolver import (
    CurrentArtifactPointer,
    CurrentPointerState,
    GitHubArtifactResolver,
)
from kapso.cross_run.security_authority_contracts import (
    SecurityDenylistObservation,
    TaskAdapterTrustObservation,
)
from kapso.cross_run.task_adapters import (
    VerifiedTaskAdapter,
    VerifiedTaskAdapterProvider,
)

if TYPE_CHECKING:
    from kapso.cross_run.expert.validation_store import (
        ExpertReleaseActivationCommitResult,
        ExpertReleasePublicationReservation,
        ExpertReleasePublicationReservationCommitResult,
        ExpertReleasePublicationStalePermit,
        ExpertValidationStore,
    )


class ExpertReleasePublicationError(ValueError):
    """Expert publication authority is missing, stale, or contradictory."""


@dataclass(frozen=True)
class ExpertReleasePublication:
    """Durable RELEASED outcome with optional telemetry from this invocation."""

    activation: ExpertReleaseActivationCommitResult
    telemetry: PublicationTelemetry | None

    def __post_init__(self) -> None:
        receipt = self.activation.receipt
        if (
            self.activation.snapshot.state.promotion_state
            is not ExpertPromotionState.RELEASED
        ):
            raise ExpertReleasePublicationError(
                "expert publication result is not durably RELEASED"
            )
        if self.telemetry is None:
            return
        record = self.telemetry.publication_record
        pointer = receipt.github_publication_pointer
        intent = receipt.github_publication_intent
        if (
            type(self.telemetry) is not PublicationTelemetry
            or record != pointer.publication_record
            or record.artifact_id != receipt.release_id
            or self.telemetry.pointer_commit_sha
            != receipt.activation_witness.activation_commit_sha
            or self.telemetry.expected_parent_sha != intent.expected_parent_sha
            or self.telemetry.source_commit_sha != intent.source_commit_sha
            or self.telemetry.source_tree_digest != intent.source_tree_digest
            or self.telemetry.validation_closure_ids != intent.validation_closure_ids
        ):
            raise ExpertReleasePublicationError(
                "expert publication telemetry differs from its activation"
            )


class ExpertReleasePublicationGate:
    """Keep one frozen approval and predecessor stable through activation."""

    def __init__(
        self,
        publisher: ExpertReleasePublisher,
        reservation: ExpertReleasePublicationReservation,
    ) -> None:
        self.publisher = publisher
        self.reservation = reservation
        self.expected_activation_predecessor_pointer: CurrentArtifactPointer | None = (
            None
        )
        self.preflight_mode: str | None = None

    def validate_before_publication(
        self,
        *,
        envelope: PublicationEnvelope,
        repositories: ScopeRepositorySettings,
        current_state: CurrentPointerState,
        manifest: ExpertBaseReleaseManifest,
        source_tree_digest: str,
        manifest_digest: str,
    ) -> None:
        if (
            self.preflight_mode is not None
            or type(self.publisher) is not ExpertReleasePublisher
            or type(manifest) is not ExpertBaseReleaseManifest
        ):
            raise ExpertReleasePublicationError(
                "expert publication gate input is invalid"
            )
        reservation = self.publisher._require_reservation(self.reservation)
        plan = reservation.plan
        self._validate_envelope(
            envelope=envelope,
            repositories=repositories,
            manifest=manifest,
            source_tree_digest=source_tree_digest,
            manifest_digest=manifest_digest,
            reservation=reservation,
        )
        pointer = current_state.pointer
        if pointer == plan.activation_predecessor_pointer:
            if current_state.head_commit_sha != (
                plan.current_release_observation.default_branch_head_commit_sha
            ):
                raise ExpertReleasePublicationError(
                    "expert activation predecessor is not at its reserved stable head"
                )
            observed = self.publisher._refresh_publication_authority(
                reservation=reservation,
                activation_pointer=None,
            )
            self._validate_observed_current(
                plan,
                repositories,
                current_state,
                observed,
            )
            self.expected_activation_predecessor_pointer = (
                plan.activation_predecessor_pointer
            )
            self.preflight_mode = "activation-predecessor"
            return
        if pointer is not None and (
            pointer.publication_record.artifact_id == plan.release_id
        ):
            observed = self.publisher.current_release_authority.observe_task_evaluation_current(
                plan.scope_id
            )
            self._validate_observed_current(
                plan,
                repositories,
                current_state,
                observed,
            )
            self._validate_release_pointer(plan, pointer)
            self.preflight_mode = "active-release"
            return
        raise ExpertReleasePublicationError(
            "expert CURRENT is neither the reserved activation predecessor nor "
            "release"
        )

    def revalidate_before_activation(
        self,
        *,
        envelope: PublicationEnvelope,
        repositories: ScopeRepositorySettings,
        pointer: CurrentArtifactPointer,
        manifest: ExpertBaseReleaseManifest,
    ) -> None:
        if self.preflight_mode != "activation-predecessor":
            raise ExpertReleasePublicationError(
                "expert activation was not preauthorized from its predecessor"
            )
        reservation = self.publisher._require_reservation(self.reservation)
        plan = reservation.plan
        if manifest != reservation.manifest:
            raise ExpertReleasePublicationError(
                "expert activation manifest differs from its reservation"
            )
        self._validate_release_pointer(plan, pointer)
        current_state = self.publisher.resolver.read_current_pointer_state(
            plan.scope_id,
            PublicationArtifactKind.EXPERT_BASE_RELEASE,
            allow_missing=True,
        )
        if (
            current_state.pointer != self.expected_activation_predecessor_pointer
            or current_state.head_commit_sha != envelope.expected_parent_sha
        ):
            raise ExpertReleasePublicationError(
                "expert CURRENT changed before release activation"
            )
        observed = self.publisher._refresh_publication_authority(
            reservation=reservation,
            activation_pointer=pointer,
        )
        self._validate_observed_current(plan, repositories, current_state, observed)

    @staticmethod
    def _validate_envelope(
        *,
        envelope: PublicationEnvelope,
        repositories: ScopeRepositorySettings,
        manifest: ExpertBaseReleaseManifest,
        source_tree_digest: str,
        manifest_digest: str,
        reservation: ExpertReleasePublicationReservation,
    ) -> None:
        plan = reservation.plan
        assets = tuple(
            (
                asset.name,
                asset.media_type,
                asset.size,
                asset.sha256,
            )
            for asset in envelope.assets
        )
        planned_assets = tuple(
            (
                asset.name,
                asset.media_type,
                asset.size,
                asset.sha256,
            )
            for asset in plan.assets
        )
        if (
            envelope.artifact_kind is not PublicationArtifactKind.EXPERT_BASE_RELEASE
            or envelope.artifact_id != plan.release_id
            or envelope.scope_id != plan.scope_id
            or envelope.expected_parent_sha
            != plan.current_release_observation.default_branch_head_commit_sha
            or envelope.manifest_relative_path != EXPERT_RELEASE_MANIFEST_PATH
            or envelope.tag != plan.tag
            or envelope.committed_at != reservation.intent.committed_at
            or envelope.validation_closure_ids != plan.validation_closure_ids
            or repositories.scope_id != plan.scope_id
            or repositories.expert_repository
            != plan.current_release_observation.repository_full_name
            or manifest != reservation.manifest
            or manifest_digest != plan.manifest_digest
            or source_tree_digest != plan.publication_source_tree_digest
            or assets != planned_assets
        ):
            raise ExpertReleasePublicationError(
                "expert publication envelope differs from its reservation"
            )

    @staticmethod
    def _validate_observed_current(
        plan: ExpertReleasePublicationPlan,
        repositories: ScopeRepositorySettings,
        current_state: CurrentPointerState,
        observed: TaskEvaluationCurrentReleaseObservation,
    ) -> None:
        pointer = current_state.pointer
        if type(observed) is not TaskEvaluationCurrentReleaseObservation:
            raise ExpertReleasePublicationError(
                "expert publication CURRENT observation is not exact"
            )
        if pointer is None:
            pointer_release_id = None
            pointer_publication_id = None
            pointer_digest = None
            pointer_closure: tuple[str, ...] = ()
        else:
            pointer_release_id = pointer.publication_record.artifact_id
            pointer_publication_id = pointer.publication_record.publication_id
            pointer_digest = tree_or_blob_digest(pointer.to_json_bytes())
            pointer_closure = pointer.validation_closure_ids
        planned = plan.current_release_observation
        if (
            observed.scope_id != plan.scope_id
            or observed.repository_full_name != repositories.expert_repository
            or observed.repository_full_name != planned.repository_full_name
            or observed.repository_node_id != planned.repository_node_id
            or observed.default_branch_head_commit_sha != current_state.head_commit_sha
            or observed.release_id != pointer_release_id
            or observed.publication_id != pointer_publication_id
            or observed.current_pointer_digest != pointer_digest
            or observed.validation_closure_ids != pointer_closure
        ):
            raise ExpertReleasePublicationError(
                "expert CURRENT changed during publication authentication"
            )

    @staticmethod
    def _validate_release_pointer(
        plan: ExpertReleasePublicationPlan,
        pointer: CurrentArtifactPointer,
    ) -> None:
        record = pointer.publication_record
        assets = tuple(
            (
                asset.name,
                asset.media_type,
                asset.size,
                asset.sha256,
            )
            for asset in record.assets
        )
        planned_assets = tuple(
            (
                asset.name,
                asset.media_type,
                asset.size,
                asset.sha256,
            )
            for asset in plan.assets
        )
        if (
            pointer.scope_id != plan.scope_id
            or record.artifact_kind is not PublicationArtifactKind.EXPERT_BASE_RELEASE
            or record.artifact_id != plan.release_id
            or record.repository_full_name
            != plan.current_release_observation.repository_full_name
            or record.repository_node_id
            != plan.current_release_observation.repository_node_id
            or record.tag != plan.tag
            or assets != planned_assets
            or pointer.source_tree_digest != plan.publication_source_tree_digest
            or pointer.manifest_relative_path != EXPERT_RELEASE_MANIFEST_PATH
            or pointer.manifest_digest != plan.manifest_digest
            or pointer.validation_closure_ids != plan.validation_closure_ids
        ):
            raise ExpertReleasePublicationError(
                "expert release pointer differs from its publication plan"
            )


class ExpertReleasePublisher:
    """Own the exact local and GitHub authorities for expert publication."""

    __slots__ = (
        "assembler",
        "validation_store",
        "github_publisher",
        "resolver",
        "current_release_authority",
        "task_adapter_authority",
        "security_denylist_authority",
        "_eligibility_coordinator",
        "_github_client",
        "_package_validator",
        "_scope_registry",
    )

    def __init__(
        self,
        *,
        assembler: ExpertReleaseAssembler,
        validation_store: ExpertValidationStore,
        github_publisher: AutonomousGitHubPublisher,
        resolver: GitHubArtifactResolver,
        current_release_authority: GitHubExpertCurrentReleaseProvider,
        task_adapter_authority: VerifiedTaskAdapterProvider,
        security_denylist_authority: ExpertPublicationSecurityDenylistAuthority,
    ) -> None:
        publication_eligibility_authority = (
            validation_store._publication_eligibility_coordinator
        )
        if (
            type(assembler) is not ExpertReleaseAssembler
            or assembler.validation_store is not validation_store
            or type(github_publisher) is not AutonomousGitHubPublisher
            or type(resolver) is not GitHubArtifactResolver
            or github_publisher.resolver is not resolver
            or github_publisher.client is not resolver.client
            or github_publisher.settings != assembler.github_settings
            or resolver.settings != assembler.github_settings
            or type(github_publisher.package_validator)
            is not GitHubArtifactMaterializer
            or github_publisher.package_validator.client is not resolver.client
            or github_publisher.package_validator.settings != assembler.github_settings
            or type(current_release_authority) is not GitHubExpertCurrentReleaseProvider
            or current_release_authority.resolver is not resolver
            or validation_store.reducer.current_release_provider
            is not current_release_authority
            or validation_store.reducer.task_adapter_provider
            is not task_adapter_authority
            or type(publication_eligibility_authority)
            is not ExpertPublicationEligibilityCoordinator
            or publication_eligibility_authority.validation_store
            is not validation_store
            or publication_eligibility_authority.task_adapter_authority
            is not task_adapter_authority
            or publication_eligibility_authority.security_denylist_authority
            is not security_denylist_authority
        ):
            raise ExpertReleasePublicationError(
                "expert publisher authorities are not one concrete trust boundary"
            )
        object.__setattr__(self, "assembler", assembler)
        object.__setattr__(self, "validation_store", validation_store)
        object.__setattr__(self, "github_publisher", github_publisher)
        object.__setattr__(self, "resolver", resolver)
        object.__setattr__(
            self,
            "current_release_authority",
            current_release_authority,
        )
        object.__setattr__(self, "task_adapter_authority", task_adapter_authority)
        object.__setattr__(
            self,
            "security_denylist_authority",
            security_denylist_authority,
        )
        object.__setattr__(
            self,
            "_eligibility_coordinator",
            publication_eligibility_authority,
        )
        object.__setattr__(self, "_github_client", github_publisher.client)
        object.__setattr__(
            self,
            "_package_validator",
            github_publisher.package_validator,
        )
        object.__setattr__(self, "_scope_registry", resolver.scope_registry)
        self.github_publisher._bind_activation_verifier(
            PublicationArtifactKind.EXPERT_BASE_RELEASE,
            ExpertReleasePublicationGate,
        )
        self.validation_store._bind_release_publisher_authority(self)

    def __setattr__(self, name: str, value: object) -> None:
        raise ExpertReleasePublicationError("expert publisher authority is immutable")

    def reserve(
        self,
        *,
        candidate_id: str,
        committed_at: str,
    ) -> ExpertReleasePublicationReservationCommitResult:
        """Derive and freeze one publication plan from authenticated CURRENT."""

        require_content_id(candidate_id, "expert publication candidate_id")
        normalize_utc_timestamp(committed_at, "expert publication committed_at")
        self._require_bound_authority()
        if self.validation_store.reopen_release_revocation(candidate_id) is not None:
            raise ExpertReleasePublicationError("expert release is revoked")
        if self.validation_store.reopen_release_activation(candidate_id) is not None:
            raise ExpertReleasePublicationError("expert release is already active")
        package = self.assembler.build(candidate_id=candidate_id)
        durable = self.validation_store.reopen_release_publication(candidate_id)
        if durable is None:
            fence = self._approved_publication_fence(package)
            frozen_current = fence.current_release_observation
            observed_before = (
                self.current_release_authority.observe_task_evaluation_current(
                    package.manifest.scope_id
                )
            )
            current_state = self.resolver.read_current_pointer_state(
                package.manifest.scope_id,
                PublicationArtifactKind.EXPERT_BASE_RELEASE,
                allow_missing=True,
            )
            observed_after = (
                self.current_release_authority.observe_task_evaluation_current(
                    package.manifest.scope_id
                )
            )
            self._validate_planning_current(
                package=package,
                frozen_current=frozen_current,
                observed_before=observed_before,
                current_state=current_state,
                observed_after=observed_after,
            )
            plan = self.assembler._derive_publication_plan(
                package=package,
                current_release_observation=frozen_current,
                activation_predecessor_pointer=current_state.pointer,
            )
        else:
            plan = self.assembler._derive_publication_plan(
                package=package,
                current_release_observation=(
                    durable.plan.current_release_observation
                ),
                activation_predecessor_pointer=(
                    durable.plan.activation_predecessor_pointer
                ),
            )
            if plan != durable.plan or package.manifest != durable.manifest:
                raise ExpertReleasePublicationError(
                    "durable expert publication reservation is not reproducible"
                )
        return self.validation_store._reserve_release_publication(
            self,
            plan=plan,
            package=package,
            committed_at=committed_at,
        )

    def publish(
        self,
        *,
        candidate_id: str,
        committed_at: str,
    ) -> ExpertReleasePublication:
        """Reserve, resume, or complete one immutable publication transaction."""

        require_content_id(candidate_id, "expert publication candidate_id")
        normalize_utc_timestamp(committed_at, "expert publication committed_at")
        self._require_bound_authority()
        if self.validation_store.reopen_release_revocation(candidate_id) is not None:
            raise ExpertReleasePublicationError("expert release is revoked")
        durable = self.validation_store.reopen_release_activation(candidate_id)
        if durable is not None:
            return ExpertReleasePublication(
                activation=durable,
                telemetry=None,
            )
        reservation = self.reserve(
            candidate_id=candidate_id,
            committed_at=committed_at,
        ).reservation
        recovered = self._recover_release_activation(reservation)
        if recovered is not None:
            return ExpertReleasePublication(activation=recovered, telemetry=None)
        package = self.assembler.build(candidate_id=candidate_id)
        with tempfile.TemporaryDirectory(prefix="kapso-expert-release-") as root:
            release_root = Path(root)
            source_tree = release_root / "source"
            asset_root = release_root / "assets"
            source_tree.mkdir(mode=0o700)
            asset_root.mkdir(mode=0o700)
            self._write_source_tree(package, source_tree)
            assets = self._write_assets(package, reservation.plan, asset_root)
            envelope = PublicationEnvelope(
                artifact_kind=PublicationArtifactKind.EXPERT_BASE_RELEASE,
                artifact_id=reservation.plan.release_id,
                scope_id=reservation.plan.scope_id,
                expected_parent_sha=(
                    reservation.plan.current_release_observation.default_branch_head_commit_sha
                ),
                source_tree=source_tree,
                manifest_relative_path=EXPERT_RELEASE_MANIFEST_PATH,
                assets=assets,
                tag=reservation.plan.tag,
                committed_at=reservation.intent.committed_at,
                validation_closure_ids=reservation.plan.validation_closure_ids,
            )
            telemetry = self.github_publisher.publish(
                envelope,
                activation_authorization=self.github_publisher._authorize_publication(
                    envelope,
                    ExpertReleasePublicationGate(self, reservation),
                ),
            )
        activation = self.complete_release(candidate_id=candidate_id)
        return ExpertReleasePublication(activation=activation, telemetry=telemetry)

    def _approved_publication_fence(
        self,
        package: ExpertReleasePackage,
    ) -> ExpertPublicationEligibilityAuthorityFence:
        snapshot = self.validation_store.snapshot(package.manifest.candidate_id)
        result = (
            None
            if snapshot is None or not snapshot.accepted_stage_results
            else snapshot.accepted_stage_results[-1]
        )
        if (
            snapshot is None
            or snapshot.state.promotion_state is not ExpertPromotionState.APPROVED
            or type(result) is not ExpertPublicationEligibilityStageResultRecord
            or result.stage_result_record_id
            != package.manifest.publication_eligibility_result_id
            or type(result.publication_authority_fence)
            is not ExpertPublicationEligibilityAuthorityFence
        ):
            raise ExpertReleasePublicationError(
                "expert publication package lacks its approved CURRENT fence"
            )
        return result.publication_authority_fence

    def _validate_planning_current(
        self,
        *,
        package: ExpertReleasePackage,
        frozen_current: TaskEvaluationCurrentReleaseObservation,
        observed_before: TaskEvaluationCurrentReleaseObservation,
        current_state: CurrentPointerState,
        observed_after: TaskEvaluationCurrentReleaseObservation,
    ) -> None:
        manifest = package.manifest
        if (
            type(current_state) is not CurrentPointerState
            or type(frozen_current) is not TaskEvaluationCurrentReleaseObservation
            or type(observed_before) is not TaskEvaluationCurrentReleaseObservation
            or type(observed_after) is not TaskEvaluationCurrentReleaseObservation
        ):
            raise ExpertReleasePublicationError(
                "expert publication planning authority is not exact"
            )
        repositories = self.resolver.repositories_for_scope(manifest.scope_id)
        pointer = current_state.pointer
        if pointer is None:
            state_release_id = None
            state_publication_id = None
            state_pointer_digest = None
            state_validation_closure_ids: tuple[str, ...] = ()
            state_repository_node_id = frozen_current.repository_node_id
        else:
            record = pointer.publication_record
            state_release_id = record.artifact_id
            state_publication_id = record.publication_id
            state_pointer_digest = tree_or_blob_digest(pointer.to_json_bytes())
            state_validation_closure_ids = pointer.validation_closure_ids
            state_repository_node_id = record.repository_node_id
            if (
                pointer.scope_id != manifest.scope_id
                or record.artifact_kind
                is not PublicationArtifactKind.EXPERT_BASE_RELEASE
                or record.repository_full_name != repositories.expert_repository
            ):
                raise ExpertReleasePublicationError(
                    "expert publication predecessor pointer has another authority"
                )
        if (
            observed_before != frozen_current
            or observed_after != frozen_current
            or frozen_current.scope_id != manifest.scope_id
            or frozen_current.release_id
            != manifest.lineage.activation_predecessor_release_id
            or frozen_current.release_id != state_release_id
            or frozen_current.publication_id != state_publication_id
            or frozen_current.repository_full_name
            != repositories.expert_repository
            or frozen_current.repository_node_id != state_repository_node_id
            or frozen_current.default_branch_head_commit_sha
            != current_state.head_commit_sha
            or frozen_current.current_pointer_digest != state_pointer_digest
            or frozen_current.validation_closure_ids
            != state_validation_closure_ids
        ):
            raise ExpertReleasePublicationError(
                "expert publication CURRENT changed after approval"
            )

    def complete_release(
        self,
        *,
        candidate_id: str,
    ) -> ExpertReleaseActivationCommitResult:
        """Recover and durably record an active or historically active release."""

        require_content_id(candidate_id, "expert release activation candidate_id")
        self._require_bound_authority()
        durable = self.validation_store.reopen_release_activation(candidate_id)
        if durable is not None:
            return durable
        reservation = self.validation_store.reopen_release_publication(candidate_id)
        if reservation is None:
            raise ExpertReleasePublicationError(
                "expert release has no pending or durable activation"
            )
        recovered = self._recover_release_activation(reservation)
        if recovered is None:
            raise ExpertReleasePublicationError(
                "prepared expert activation has not won CURRENT"
            )
        return recovered

    def _recover_release_activation(
        self,
        reservation: ExpertReleasePublicationReservation,
    ) -> ExpertReleaseActivationCommitResult | None:
        plan = reservation.plan
        artifact_kind = PublicationArtifactKind.EXPERT_BASE_RELEASE
        github_intent = self.resolver.read_artifact_intent(
            plan.scope_id,
            artifact_kind,
            plan.release_id,
        )
        github_pointer = self.resolver.read_artifact_pointer(
            plan.scope_id,
            artifact_kind,
            plan.release_id,
        )
        self.validation_store._validate_release_publication_remote_history(
            reservation.intent,
            plan,
            github_intent,
            github_pointer,
            self.github_publisher.settings.publisher_login,
        )
        if github_intent is None or github_pointer is None:
            return None
        resolved = self.resolver.resolve_artifact(
            plan.scope_id,
            artifact_kind,
            plan.release_id,
        )
        if resolved.pointer != github_pointer:
            raise ExpertReleasePublicationError(
                "expert release identity changed during activation recovery"
            )
        activation_commit_sha = self.resolver.resolve_artifact_activation_preparation(
            plan.scope_id,
            artifact_kind,
            plan.release_id,
            github_intent,
            github_pointer,
            allow_missing=True,
        )
        if activation_commit_sha is None:
            return None
        activation_witness = self.resolver.resolve_artifact_activation_witness(
            plan.scope_id,
            artifact_kind,
            plan.release_id,
            github_intent,
            github_pointer,
            allow_missing=True,
        )
        observed = self.current_release_authority.observe_task_evaluation_current(
            plan.scope_id
        )
        if activation_witness is None and (
            observed.release_id == plan.release_id
            and observed.publication_id
            == github_pointer.publication_record.publication_id
            and observed.current_pointer_digest
            == tree_or_blob_digest(github_pointer.to_json_bytes())
            and observed.default_branch_head_commit_sha == activation_commit_sha
        ):
            activation_witness = (
                self.github_publisher.finalize_artifact_activation_witness(
                    plan.scope_id,
                    artifact_kind,
                    plan.release_id,
                    github_intent,
                    github_pointer,
                )
            )
        if activation_witness is None:
            activation_witness = self.resolver.resolve_artifact_activation_witness(
                plan.scope_id,
                artifact_kind,
                plan.release_id,
                github_intent,
                github_pointer,
                allow_missing=True,
            )
            if activation_witness is None:
                return None
        self.resolver.require_artifact_intent(
            plan.scope_id,
            artifact_kind,
            plan.release_id,
            github_intent,
        )
        self.resolver.require_artifact_pointer(
            plan.scope_id,
            artifact_kind,
            plan.release_id,
            github_pointer,
        )
        if (
            self.resolver.resolve_artifact_activation_preparation(
                plan.scope_id,
                artifact_kind,
                plan.release_id,
                github_intent,
                github_pointer,
            )
            != activation_commit_sha
        ):
            raise ExpertReleasePublicationError(
                "expert activation identity changed during recovery"
            )
        if (
            self.resolver.resolve_artifact_activation_witness(
                plan.scope_id,
                artifact_kind,
                plan.release_id,
                github_intent,
                github_pointer,
            )
            != activation_witness
        ):
            raise ExpertReleasePublicationError(
                "expert activation witness changed during recovery"
            )
        refreshed = self.current_release_authority.observe_task_evaluation_current(
            plan.scope_id
        )
        if refreshed != observed:
            raise ExpertReleasePublicationError(
                "expert CURRENT changed during activation recovery"
            )
        permit = self.validation_store._seal_release_activation(
            publisher=self,
            reservation=reservation,
            github_publication_intent=github_intent,
            github_publication_pointer=github_pointer,
            activation_witness=activation_witness,
            observed_current=refreshed,
        )
        return self.validation_store.commit_release_activation(permit)

    def _require_reservation(
        self,
        reservation: ExpertReleasePublicationReservation,
    ) -> ExpertReleasePublicationReservation:
        self._require_bound_authority()
        current = self.validation_store.reopen_release_publication(
            reservation.plan.candidate_id
        )
        if current != reservation:
            raise ExpertReleasePublicationError(
                "expert publication reservation is no longer active"
            )
        return current

    def _require_bound_authority(self) -> None:
        self._require_local_authority_join()
        self.validation_store._require_bound_release_publisher_authority(self)

    def _require_local_authority_join(self) -> None:
        coordinator = self.validation_store._publication_eligibility_coordinator
        package_validator = self.github_publisher.package_validator
        if (
            type(self.assembler) is not ExpertReleaseAssembler
            or self.assembler.validation_store is not self.validation_store
            or type(self.github_publisher) is not AutonomousGitHubPublisher
            or type(self.resolver) is not GitHubArtifactResolver
            or self.resolver.scope_registry is not self._scope_registry
            or self.github_publisher.resolver is not self.resolver
            or self.github_publisher.client is not self._github_client
            or self.resolver.client is not self._github_client
            or self.github_publisher.settings != self.assembler.github_settings
            or self.resolver.settings != self.assembler.github_settings
            or type(package_validator) is not GitHubArtifactMaterializer
            or package_validator is not self._package_validator
            or package_validator.client is not self._github_client
            or package_validator.settings != self.assembler.github_settings
            or type(self.current_release_authority)
            is not GitHubExpertCurrentReleaseProvider
            or self.current_release_authority.resolver is not self.resolver
            or self.validation_store.reducer.current_release_provider
            is not self.current_release_authority
            or self.validation_store.reducer.task_adapter_provider
            is not self.task_adapter_authority
            or type(coordinator) is not ExpertPublicationEligibilityCoordinator
            or coordinator is not self._eligibility_coordinator
            or coordinator.validation_store is not self.validation_store
            or coordinator.task_adapter_authority is not self.task_adapter_authority
            or coordinator.security_denylist_authority
            is not self.security_denylist_authority
            or self.github_publisher._activation_verifier_types.get(
                PublicationArtifactKind.EXPERT_BASE_RELEASE
            )
            is not ExpertReleasePublicationGate
        ):
            raise ExpertReleasePublicationError(
                "expert publisher authority binding changed after construction"
            )

    def _refresh_publication_authority(
        self,
        *,
        reservation: ExpertReleasePublicationReservation,
        activation_pointer: CurrentArtifactPointer | None,
    ) -> TaskEvaluationCurrentReleaseObservation:
        reservation = self._require_reservation(reservation)
        plan = reservation.plan
        fence = self._publication_eligibility_fence(reservation)
        current_before = self.current_release_authority.observe_task_evaluation_current(
            plan.scope_id
        )
        if current_before != plan.current_release_observation:
            raise ExpertReleasePublicationError(
                "expert publication CURRENT differs from its approved authority"
            )
        adapter_observations = self._reverify_task_adapters(fence)
        security_subject_ids = self._publication_security_subject_ids(
            reservation=reservation,
            fence=fence,
            current=current_before,
            adapter_observations=adapter_observations,
            activation_pointer=activation_pointer,
        )
        denylist = self.security_denylist_authority.observe_exact(
            scope_id=plan.scope_id,
            scope_contract_id=plan.scope_contract_id,
            checked_subject_ids=security_subject_ids,
        )
        repositories = self.resolver.repositories_for_scope(plan.scope_id)
        if (
            type(denylist) is not SecurityDenylistObservation
            or denylist.scope_id != plan.scope_id
            or denylist.scope_contract_id != plan.scope_contract_id
            or denylist.scope_repository_binding_hash
            != repositories.binding_fingerprint
            or denylist.repository_full_name != repositories.security_repository
            or denylist.checked_subject_ids != security_subject_ids
            or denylist.matched_revocations
        ):
            raise ExpertReleasePublicationError(
                "expert publication denylist differs from fresh authority"
            )
        current_after = self.current_release_authority.observe_task_evaluation_current(
            plan.scope_id
        )
        if current_after != current_before:
            raise ExpertReleasePublicationError(
                "expert CURRENT changed during publication authority refresh"
            )
        self._require_reservation(reservation)
        return current_after

    def _reverify_task_adapters(
        self,
        fence: ExpertPublicationEligibilityAuthorityFence,
    ) -> tuple[TaskAdapterTrustObservation, ...]:
        observations: list[TaskAdapterTrustObservation] = []
        for expected in fence.task_adapter_trust_observations:
            verified = self.task_adapter_authority.resolve_exact(
                task_adapter_manifest_id=expected.task_adapter_manifest_id,
                verification_receipt_id=expected.verification_receipt_id,
            )
            if type(verified) is not VerifiedTaskAdapter:
                raise ExpertReleasePublicationError(
                    "expert publication adapter authority is not exact"
                )
            observed = TaskAdapterTrustObservation.mint(
                task_adapter_manifest_id=(verified.manifest.task_adapter_manifest_id),
                verification_receipt_id=(
                    verified.verification_receipt.verification_receipt_id
                ),
                verifier_id=verified.verification_receipt.verifier_id,
                verifier_version=verified.verification_receipt.verifier_version,
                dependency_ids=verified.dependency_ids,
            )
            if observed != expected:
                raise ExpertReleasePublicationError(
                    "expert publication adapter differs from approved authority"
                )
            observations.append(observed)
        ordered = tuple(sorted(observations, key=lambda item: item.observation_id))
        if ordered != fence.task_adapter_trust_observations:
            raise ExpertReleasePublicationError(
                "expert publication adapter observations are not canonical"
            )
        return ordered

    @staticmethod
    def _publication_eligibility_fence(
        reservation: ExpertReleasePublicationReservation,
    ) -> ExpertPublicationEligibilityAuthorityFence:
        result = reservation.snapshot.accepted_stage_results[-1]
        if (
            type(result) is not ExpertPublicationEligibilityStageResultRecord
            or result.stage_result_record_id
            != reservation.plan.publication_eligibility_result_id
            or type(result.publication_authority_fence)
            is not ExpertPublicationEligibilityAuthorityFence
        ):
            raise ExpertReleasePublicationError(
                "expert reservation lacks publication eligibility authority"
            )
        return result.publication_authority_fence

    @staticmethod
    def _publication_security_subject_ids(
        *,
        reservation: ExpertReleasePublicationReservation,
        fence: ExpertPublicationEligibilityAuthorityFence,
        current: TaskEvaluationCurrentReleaseObservation,
        adapter_observations: tuple[TaskAdapterTrustObservation, ...],
        activation_pointer: CurrentArtifactPointer | None,
    ) -> tuple[str, ...]:
        plan = reservation.plan
        prior_denylist = fence.security_denylist_observation
        subjects = {
            reservation.intent.publication_intent_id,
            plan.publication_plan_id,
            plan.release_id,
            plan.candidate_id,
            plan.approval_transition_id,
            plan.approval_state_id,
            plan.publication_eligibility_result_id,
            fence.fence_id,
            prior_denylist.observation_id,
            prior_denylist.snapshot_id,
            prior_denylist.publication_id,
            current.observation_id,
            *plan.validation_closure_ids,
            *fence.security_subject_ids,
            *current.validation_closure_ids,
        }
        if current.release_id is not None:
            subjects.add(current.release_id)
        if current.publication_id is not None:
            subjects.add(current.publication_id)
        if activation_pointer is not None:
            subjects.add(activation_pointer.publication_record.publication_id)
        for observation in adapter_observations:
            subjects.update(
                {
                    observation.observation_id,
                    observation.task_adapter_manifest_id,
                    observation.verification_receipt_id,
                    observation.verifier_authority_subject_id,
                    *observation.dependency_ids,
                }
            )
        ordered = tuple(sorted(subjects))
        for subject_id in ordered:
            require_content_id(subject_id, "expert publication security subject")
        return ordered

    @staticmethod
    def _write_source_tree(
        package: ExpertReleasePackage,
        source_tree: Path,
    ) -> None:
        for relative_path, (payload, mode) in package.publication_files.items():
            path = PurePosixPath(relative_path)
            if (
                path.is_absolute()
                or ".." in path.parts
                or path.as_posix() != relative_path
                or mode not in {"100644", "100755"}
            ):
                raise ExpertReleasePublicationError(
                    "expert publication source path or mode is invalid"
                )
            destination = source_tree.joinpath(*path.parts)
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            ExpertReleasePublisher._write_file(
                destination,
                payload,
                0o755 if mode == "100755" else 0o644,
            )

    @staticmethod
    def _write_assets(
        package: ExpertReleasePackage,
        plan: ExpertReleasePublicationPlan,
        asset_root: Path,
    ) -> tuple[ReleaseAssetInput, ...]:
        payloads = {
            EXPERT_RELEASE_CONTROL_ARCHIVE: package.control_archive,
            EXPERT_RELEASE_EVIDENCE_ARCHIVE: package.evidence_archive,
            EXPERT_RELEASE_SOURCE_ARCHIVE: package.source_archive,
        }
        assets: list[ReleaseAssetInput] = []
        for descriptor in plan.assets:
            payload = payloads.get(descriptor.name)
            if (
                payload is None
                or len(payload) != descriptor.size
                or tree_or_blob_digest(payload) != descriptor.sha256
            ):
                raise ExpertReleasePublicationError(
                    "expert publication asset differs from its plan"
                )
            path = asset_root / descriptor.name
            ExpertReleasePublisher._write_file(path, payload, 0o600)
            assets.append(
                ReleaseAssetInput(
                    path=path,
                    name=descriptor.name,
                    media_type=descriptor.media_type,
                    size=descriptor.size,
                    sha256=descriptor.sha256,
                )
            )
        if set(payloads) != {asset.name for asset in plan.assets}:
            raise ExpertReleasePublicationError(
                "expert publication plan omits a required release asset"
            )
        return tuple(assets)

    @staticmethod
    def _write_file(path: Path, payload: bytes, mode: int) -> None:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            mode,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

    def resolve_stale(
        self,
        *,
        candidate_id: str,
        resolved_at: str,
    ) -> ExpertReleasePublicationStaleResolution:
        """Resolve only an intent displaced by another active expert release."""

        require_content_id(candidate_id, "stale expert publication candidate_id")
        self._require_bound_authority()
        reservation = self.validation_store.reopen_release_publication(candidate_id)
        if reservation is None:
            return self.validation_store.reopen_stale_release_publication(candidate_id)
        plan = reservation.plan
        state = self.resolver.read_current_pointer_state(
            plan.scope_id,
            PublicationArtifactKind.EXPERT_BASE_RELEASE,
            allow_missing=True,
        )
        pointer = state.pointer
        if pointer is None:
            raise ExpertReleasePublicationError(
                "absent CURRENT cannot displace a reserved expert release"
            )
        active_release_id = pointer.publication_record.artifact_id
        if active_release_id == plan.release_id:
            raise ExpertReleasePublicationError(
                "active reserved release requires RELEASED recovery"
            )
        if active_release_id == plan.lineage.activation_predecessor_release_id:
            raise ExpertReleasePublicationError(
                "pending predecessor-preserving publication must resume before "
                "resolution"
            )
        observed = self.current_release_authority.observe_task_evaluation_current(
            plan.scope_id
        )
        if (
            observed.release_id != active_release_id
            or observed.default_branch_head_commit_sha != state.head_commit_sha
        ):
            raise ExpertReleasePublicationError(
                "expert CURRENT changed during stale publication classification"
            )
        winner_intent = self.resolver.read_artifact_intent(
            plan.scope_id,
            PublicationArtifactKind.EXPERT_BASE_RELEASE,
            active_release_id,
        )
        winner_identity = self.resolver.read_artifact_pointer(
            plan.scope_id,
            PublicationArtifactKind.EXPERT_BASE_RELEASE,
            active_release_id,
        )
        if winner_intent is None or winner_identity != pointer:
            raise ExpertReleasePublicationError(
                "active winner lacks its exact publication identity"
            )
        winner_witness = self.github_publisher.finalize_artifact_activation_witness(
            plan.scope_id,
            PublicationArtifactKind.EXPERT_BASE_RELEASE,
            active_release_id,
            winner_intent,
            winner_identity,
        )
        own_intent = self.resolver.read_artifact_intent(
            plan.scope_id,
            PublicationArtifactKind.EXPERT_BASE_RELEASE,
            plan.release_id,
        )
        own_identity = self.resolver.read_artifact_pointer(
            plan.scope_id,
            PublicationArtifactKind.EXPERT_BASE_RELEASE,
            plan.release_id,
        )
        self.validation_store._validate_release_publication_remote_history(
            reservation.intent,
            plan,
            own_intent,
            own_identity,
            self.github_publisher.settings.publisher_login,
        )
        activation_preparation_commit_sha = None
        if own_intent is not None:
            self.resolver.diagnose_repository(
                plan.scope_id,
                PublicationArtifactKind.EXPERT_BASE_RELEASE,
            )
            if own_identity is None:
                self.resolver.verify_publication_intent_source(
                    own_intent.repository_full_name,
                    own_intent,
                )
            else:
                resolved = self.resolver.resolve_artifact(
                    plan.scope_id,
                    PublicationArtifactKind.EXPERT_BASE_RELEASE,
                    plan.release_id,
                )
                if resolved.pointer != own_identity:
                    raise ExpertReleasePublicationError(
                        "expert release identity changed during stale classification"
                    )
                activation_preparation_commit_sha = (
                    self.resolver.resolve_artifact_activation_preparation(
                        plan.scope_id,
                        PublicationArtifactKind.EXPERT_BASE_RELEASE,
                        plan.release_id,
                        own_intent,
                        own_identity,
                        allow_missing=True,
                    )
                )
                if activation_preparation_commit_sha is not None:
                    own_witness = self.resolver.resolve_artifact_activation_witness(
                        plan.scope_id,
                        PublicationArtifactKind.EXPERT_BASE_RELEASE,
                        plan.release_id,
                        own_intent,
                        own_identity,
                        allow_missing=True,
                    )
                    if own_witness is not None:
                        raise ExpertReleasePublicationError(
                            "historically active release requires RELEASED recovery"
                        )
        if (
            self.resolver.read_artifact_intent(
                plan.scope_id,
                PublicationArtifactKind.EXPERT_BASE_RELEASE,
                plan.release_id,
            )
            != own_intent
            or self.resolver.read_artifact_pointer(
                plan.scope_id,
                PublicationArtifactKind.EXPERT_BASE_RELEASE,
                plan.release_id,
            )
            != own_identity
            or (
                activation_preparation_commit_sha is not None
                and self.resolver.resolve_artifact_activation_preparation(
                    plan.scope_id,
                    PublicationArtifactKind.EXPERT_BASE_RELEASE,
                    plan.release_id,
                    own_intent,
                    own_identity,
                )
                != activation_preparation_commit_sha
            )
            or self.resolver.read_artifact_intent(
                plan.scope_id,
                PublicationArtifactKind.EXPERT_BASE_RELEASE,
                active_release_id,
            )
            != winner_intent
            or self.resolver.read_artifact_pointer(
                plan.scope_id,
                PublicationArtifactKind.EXPERT_BASE_RELEASE,
                active_release_id,
            )
            != winner_identity
            or self.resolver.resolve_artifact_activation_witness(
                plan.scope_id,
                PublicationArtifactKind.EXPERT_BASE_RELEASE,
                active_release_id,
                winner_intent,
                winner_identity,
            )
            != winner_witness
        ):
            raise ExpertReleasePublicationError(
                "expert release history changed during stale classification"
            )
        refreshed_observed = (
            self.current_release_authority.observe_task_evaluation_current(
                plan.scope_id
            )
        )
        if refreshed_observed != observed:
            raise ExpertReleasePublicationError(
                "expert CURRENT changed during stale publication classification"
            )
        if (
            activation_preparation_commit_sha is not None
            and self.resolver.resolve_artifact_activation_witness(
                plan.scope_id,
                PublicationArtifactKind.EXPERT_BASE_RELEASE,
                plan.release_id,
                own_intent,
                own_identity,
                allow_missing=True,
            )
            is not None
        ):
            raise ExpertReleasePublicationError(
                "release activation witness appeared during stale classification"
            )
        permit = self.validation_store._seal_stale_release_publication(
            publisher=self,
            reservation=reservation,
            observed_current=refreshed_observed,
            observed_current_activation_witness=winner_witness,
            own_github_publication_intent=own_intent,
            own_github_publication_pointer=own_identity,
            own_github_activation_preparation_commit_sha=(
                activation_preparation_commit_sha
            ),
            resolved_at=resolved_at,
        )
        return self.validation_store.resolve_stale_release_publication(permit)


__all__ = [
    "ExpertReleasePublication",
    "ExpertReleasePublicationError",
    "ExpertReleasePublicationGate",
    "ExpertReleasePublisher",
]
