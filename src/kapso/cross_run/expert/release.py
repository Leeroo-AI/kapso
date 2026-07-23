"""Verification-only assembly of immutable expert release packages."""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    content_id,
    is_content_id,
    parse_json_bytes,
    source_tree_digest,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    ExpertBaseReleaseManifest,
    ExpertCandidateSanitationStatus,
    ExpertEvaluatorResultRecord,
    ExpertPromotionState,
    ExpertReleaseLineage,
    ExpertValidationAttempt,
    StrictContract,
)
from kapso.cross_run.expert.book import EXPERT_BOOK_PATH
from kapso.cross_run.expert.candidates import ExpertCandidateValidator
from kapso.cross_run.expert.promotion_authority_contracts import (
    ExpertCandidateReleaseUseOutcome,
    ExpertPublicationEligibilityStageResultRecord,
)
from kapso.cross_run.expert.promotion_decision_contracts import (
    ExpertReleaseMatrixDecisionOutcome,
)
from kapso.cross_run.expert.promotion_stage_contracts import (
    ExpertReleaseMatrixStageResultRecord,
)
from kapso.cross_run.expert.release_contracts import (
    ExpertReleaseAssetDescriptor,
    ExpertReleaseEvidenceManifest,
    ExpertReleaseMatrixSummary,
    ExpertReleasePublicationPlan,
)
from kapso.cross_run.expert.review_contracts import (
    ExpertAutomatedReviewStageResultRecord,
)
from kapso.cross_run.expert.store import (
    ExpertCandidateStore,
    StoredExpertCandidate,
    stored_candidate_admission_dependency_ids,
)
from kapso.cross_run.expert.task_evaluation_authority_contracts import (
    TaskEvaluationCurrentReleaseObservation,
)
from kapso.cross_run.expert.validation_snapshots import (
    ExpertValidationSnapshot,
    ExpertValidationTransition,
)
from kapso.cross_run.github.resolver import CurrentArtifactPointer
from kapso.cross_run.settings import ExpertSettings, GitHubSettings
from kapso.cross_run.source_archives import build_deterministic_tar_zst

if TYPE_CHECKING:
    from kapso.cross_run.expert.validation_store import ExpertValidationStore

EXPERT_RELEASE_MANIFEST_PATH = ".kapso/expert/release.json"
EXPERT_RELEASE_EVIDENCE_ROOT = ".kapso/expert/release-evidence"
EXPERT_RELEASE_EVIDENCE_MANIFEST_PATH = f"{EXPERT_RELEASE_EVIDENCE_ROOT}/manifest.json"
EXPERT_RELEASE_MATRIX_SUMMARY_PATH = (
    f"{EXPERT_RELEASE_EVIDENCE_ROOT}/matrix-summary.json"
)
EXPERT_RELEASE_SOURCE_ARCHIVE = "expert-source.tar.zst"
EXPERT_RELEASE_EVIDENCE_ARCHIVE = "expert-evidence.tar.zst"
EXPERT_RELEASE_CONTROL_ARCHIVE = "expert-control.tar.zst"
_RELEASE_RECORD_ROOT = f"{EXPERT_RELEASE_EVIDENCE_ROOT}/records"
_REGULAR_MODE = "100644"


class ExpertReleaseAssemblyError(ValueError):
    """An approved candidate cannot be projected as an exact release."""


@dataclass(frozen=True)
class ExpertReleasePackage:
    """Exact scientific source, safe evidence, and deterministic release carriers."""

    manifest: ExpertBaseReleaseManifest
    evidence_manifest: ExpertReleaseEvidenceManifest
    matrix_summary: ExpertReleaseMatrixSummary
    source_files: Mapping[str, tuple[bytes, str]]
    evidence_files: Mapping[str, tuple[bytes, str]]
    source_archive: bytes
    evidence_archive: bytes
    control_archive: bytes

    def __post_init__(self) -> None:
        if (
            type(self.manifest) is not ExpertBaseReleaseManifest
            or type(self.evidence_manifest) is not ExpertReleaseEvidenceManifest
            or type(self.matrix_summary) is not ExpertReleaseMatrixSummary
            or not isinstance(self.source_files, Mapping)
            or not isinstance(self.evidence_files, Mapping)
            or type(self.source_archive) is not bytes
            or type(self.evidence_archive) is not bytes
            or type(self.control_archive) is not bytes
        ):
            raise ExpertReleaseAssemblyError(
                "expert release package requires exact typed components"
            )
        object.__setattr__(
            self,
            "source_files",
            MappingProxyType(dict(sorted(self.source_files.items()))),
        )
        object.__setattr__(
            self,
            "evidence_files",
            MappingProxyType(dict(sorted(self.evidence_files.items()))),
        )
        if (
            self.manifest.evidence_manifest_ref
            != self.evidence_manifest.evidence_manifest_id
            or self.manifest.test_matrix_summary_ref != self.matrix_summary.summary_id
            or self.evidence_manifest.release_matrix_summary_id
            != self.matrix_summary.summary_id
        ):
            raise ExpertReleaseAssemblyError(
                "expert release package records differ from its manifest"
            )

    @property
    def publication_files(self) -> Mapping[str, tuple[bytes, str]]:
        files = dict(self.source_files)
        for relative_path, value in self.evidence_files.items():
            if relative_path in files:
                raise ExpertReleaseAssemblyError(
                    "expert source collides with release evidence"
                )
            files[relative_path] = value
        if EXPERT_RELEASE_MANIFEST_PATH in files:
            raise ExpertReleaseAssemblyError(
                "expert source collides with its release manifest"
            )
        files[EXPERT_RELEASE_MANIFEST_PATH] = (
            self.manifest.to_json_bytes(),
            _REGULAR_MODE,
        )
        return MappingProxyType(dict(sorted(files.items())))


class ExpertReleaseAssembler:
    """Build a release without changing one byte of the approved candidate tree."""

    def __init__(
        self,
        *,
        candidate_store: ExpertCandidateStore,
        validation_store: ExpertValidationStore,
        expert_settings: ExpertSettings,
        github_settings: GitHubSettings,
    ) -> None:
        if (
            candidate_store.validator.settings != expert_settings
            or validation_store.reducer.candidate_store is not candidate_store
        ):
            raise ExpertReleaseAssemblyError(
                "release assembler candidate authority differs from configuration"
            )
        self.candidate_store = candidate_store
        self.validation_store = validation_store
        self.candidate_validator: ExpertCandidateValidator = candidate_store.validator
        self.expert_settings = expert_settings
        self.github_settings = github_settings
        self.validation_store._bind_release_assembly_authority(self)

    def _derive_publication_plan(
        self,
        *,
        package: ExpertReleasePackage,
        current_release_observation: TaskEvaluationCurrentReleaseObservation,
        activation_predecessor_pointer: CurrentArtifactPointer | None,
    ) -> ExpertReleasePublicationPlan:
        """Derive the sole publication plan for an approved package."""

        if (
            type(package) is not ExpertReleasePackage
            or type(current_release_observation)
            is not TaskEvaluationCurrentReleaseObservation
        ):
            raise ExpertReleaseAssemblyError(
                "publication planning requires an exact package and CURRENT "
                "observation"
            )
        rebuilt = self.build(candidate_id=package.manifest.candidate_id)
        if rebuilt != package:
            raise ExpertReleaseAssemblyError(
                "publication package differs from deterministic assembly"
            )
        expected_assets = tuple(
            sorted(
                (
                    ExpertReleaseAssetDescriptor(
                        name=name,
                        media_type="application/zstd",
                        size=len(payload),
                        sha256=tree_or_blob_digest(payload),
                    )
                    for name, payload in (
                        (EXPERT_RELEASE_CONTROL_ARCHIVE, package.control_archive),
                        (EXPERT_RELEASE_EVIDENCE_ARCHIVE, package.evidence_archive),
                        (EXPERT_RELEASE_SOURCE_ARCHIVE, package.source_archive),
                    )
                ),
                key=lambda asset: asset.name,
            )
        )
        manifest = package.manifest
        if activation_predecessor_pointer is None:
            generation = 0
        else:
            predecessor_tag = activation_predecessor_pointer.publication_record.tag
            match = re.fullmatch(
                rf"{re.escape(self.github_settings.expert_tag_prefix)}E([0-9]+)",
                predecessor_tag,
            )
            if match is None:
                raise ExpertReleaseAssemblyError(
                    "activation predecessor tag differs from expert release order"
                )
            generation = int(match.group(1)) + 1
        plan = ExpertReleasePublicationPlan.mint(
            scope_contract_id=manifest.scope_contract_id,
            scope_id=manifest.scope_id,
            release_id=manifest.release_id,
            candidate_id=manifest.candidate_id,
            candidate_tree_hash=manifest.candidate_tree_hash,
            validation_attempt_id=manifest.validation_attempt_id,
            approval_transition_id=manifest.approval_transition_id,
            approval_state_id=manifest.approval_state_id,
            publication_eligibility_result_id=(
                manifest.publication_eligibility_result_id
            ),
            lineage=manifest.lineage,
            current_release_observation=current_release_observation,
            activation_predecessor_pointer=activation_predecessor_pointer,
            generation=generation,
            tag=f"{self.github_settings.expert_tag_prefix}E{generation:06d}",
            manifest_digest=tree_or_blob_digest(manifest.to_json_bytes()),
            publication_source_tree_digest=source_tree_digest(
                {
                    path: (tree_or_blob_digest(payload), mode, len(payload))
                    for path, (payload, mode) in package.publication_files.items()
                }
            ),
            assets=expected_assets,
            manifest_consumed_dependency_ids=manifest.consumed_dependency_ids,
            manifest_control_dependency_ids=manifest.control_dependency_ids,
            validation_closure_ids=tuple(
                sorted(
                    {
                        manifest.release_id,
                        *manifest.consumed_dependency_ids,
                        *manifest.control_dependency_ids,
                    }
                )
            ),
        )
        return plan

    def build(
        self,
        *,
        candidate_id: str,
    ) -> ExpertReleasePackage:
        stored_candidate = self.candidate_store.read(candidate_id)
        approved_snapshot = self.validation_store.snapshot(candidate_id)
        if approved_snapshot is None:
            raise ExpertReleaseAssemblyError(
                "release assembly requires a durable validation snapshot"
            )
        closure = stored_candidate.closure
        candidate = closure.manifest
        attempt = approved_snapshot.latest_attempt
        publication_result, matrix_result, review_result = (
            self._validate_approved_snapshot(
                stored_candidate=stored_candidate,
                snapshot=approved_snapshot,
            )
        )
        if attempt is None:
            raise ExpertReleaseAssemblyError(
                "approved release has no validation attempt"
            )
        book = self.candidate_validator.validate_persisted(closure)
        if closure.candidate_contents.get(EXPERT_BOOK_PATH) != book:
            raise ExpertReleaseAssemblyError(
                "approved semantic book differs from deterministic compilation"
            )
        if closure.sanitation_report.status is not (
            ExpertCandidateSanitationStatus.ADMITTED
        ):
            raise ExpertReleaseAssemblyError(
                "approved candidate sanitation did not pass"
            )
        source_files = self._source_files(stored_candidate)
        source_archive = self._archive(source_files)
        matrix_summary = self._matrix_summary(matrix_result, publication_result)
        records = self._evidence_records(
            stored_candidate=stored_candidate,
            snapshot=approved_snapshot,
        )
        record_files = {
            self._record_path(record): (record.to_json_bytes(), _REGULAR_MODE)
            for record in records
        }
        record_files[EXPERT_RELEASE_MATRIX_SUMMARY_PATH] = (
            matrix_summary.to_json_bytes(),
            _REGULAR_MODE,
        )
        record_ids = tuple(sorted(self._record_id(record) for record in records))
        evidence_manifest = ExpertReleaseEvidenceManifest.mint(
            candidate_id=candidate.candidate_id,
            candidate_commit_record_id=stored_candidate.commit_record.commit_record_id,
            candidate_tree_hash=candidate.candidate_tree_hash,
            validation_attempt_id=attempt.validation_attempt_id,
            approval_transition_id=approved_snapshot.transition.transition_id,
            approval_state_id=approved_snapshot.state.validation_state_id,
            publication_eligibility_result_id=(
                publication_result.stage_result_record_id
            ),
            release_matrix_summary_id=matrix_summary.summary_id,
            record_ids=record_ids,
            record_checksums={
                path: tree_or_blob_digest(payload)
                for path, (payload, _) in sorted(record_files.items())
            },
            exact_dependency_ids=tuple(
                sorted(
                    {
                        candidate.candidate_id,
                        stored_candidate.commit_record.commit_record_id,
                        attempt.validation_attempt_id,
                        approved_snapshot.transition.transition_id,
                        approved_snapshot.state.validation_state_id,
                        publication_result.stage_result_record_id,
                        matrix_summary.summary_id,
                        *record_ids,
                    }
                )
            ),
        )
        evidence_files = dict(record_files)
        evidence_files[EXPERT_RELEASE_EVIDENCE_MANIFEST_PATH] = (
            evidence_manifest.to_json_bytes(),
            _REGULAR_MODE,
        )
        evidence_archive = self._archive(evidence_files)
        evidence_dependency_ids = self._evidence_dependency_ids(evidence_files)
        direct_dependencies = {
            candidate.scope_contract_id,
            candidate.candidate_id,
            stored_candidate.commit_record.commit_record_id,
            candidate.candidate_tree_ref,
            candidate.derivation_ref,
            candidate.validation_context_ref,
            candidate.patch_ref,
            candidate.sanitation_report_id,
            candidate.proposed_repository_map_ref,
            *candidate.module_contract_refs,
            *candidate.ancestor_candidate_ids,
            *candidate.source_dependency_ids,
            *candidate.consumed_expert_release_ids,
            attempt.validation_attempt_id,
            approved_snapshot.transition.transition_id,
            approved_snapshot.state.validation_state_id,
            publication_result.stage_result_record_id,
            matrix_result.stage_result_record_id,
            matrix_result.release_matrix_report.release_matrix_report_id,
            publication_result.promotion_decision.promotion_decision_id,
            *review_result.assertion_ids,
            attempt.validation_policy_id,
            evidence_manifest.evidence_manifest_id,
            matrix_summary.summary_id,
        }
        if candidate.source_base_release_id is not None:
            direct_dependencies.add(candidate.source_base_release_id)
        manifest = ExpertBaseReleaseManifest.mint(
            scope_contract_id=candidate.scope_contract_id,
            scope_id=closure.validation_context.scope_id,
            lineage=ExpertReleaseLineage(
                source_base_release_id=candidate.source_base_release_id,
                activation_predecessor_release_id=(
                    publication_result.expected_current_release_id
                ),
            ),
            candidate_id=candidate.candidate_id,
            candidate_commit_record_id=stored_candidate.commit_record.commit_record_id,
            candidate_tree_ref=candidate.candidate_tree_ref,
            candidate_tree_hash=candidate.candidate_tree_hash,
            candidate_derivation_ref=candidate.derivation_ref,
            candidate_validation_context_ref=candidate.validation_context_ref,
            candidate_patch_ref=candidate.patch_ref,
            candidate_sanitation_report_id=candidate.sanitation_report_id,
            candidate_ancestor_ids=candidate.ancestor_candidate_ids,
            candidate_source_dependency_ids=candidate.source_dependency_ids,
            candidate_consumed_expert_release_ids=(
                candidate.consumed_expert_release_ids
            ),
            repository_map_ref=candidate.proposed_repository_map_ref,
            module_contract_refs=candidate.module_contract_refs,
            module_versions={
                module.module_id: module.version for module in closure.module_contracts
            },
            semantic_book_digest=candidate.semantic_book_digest,
            validation_attempt_id=attempt.validation_attempt_id,
            approval_transition_id=approved_snapshot.transition.transition_id,
            approval_state_id=approved_snapshot.state.validation_state_id,
            publication_eligibility_result_id=(
                publication_result.stage_result_record_id
            ),
            release_matrix_stage_result_id=matrix_result.stage_result_record_id,
            release_matrix_report_id=(
                matrix_result.release_matrix_report.release_matrix_report_id
            ),
            promotion_decision_id=(
                publication_result.promotion_decision.promotion_decision_id
            ),
            approval_assertion_ids=review_result.assertion_ids,
            validation_policy_id=attempt.validation_policy_id,
            configuration_fingerprint=attempt.configuration_fingerprint,
            source_archive_ref=EXPERT_RELEASE_SOURCE_ARCHIVE,
            evidence_archive_ref=EXPERT_RELEASE_EVIDENCE_ARCHIVE,
            evidence_manifest_ref=evidence_manifest.evidence_manifest_id,
            test_matrix_summary_ref=matrix_summary.summary_id,
            evidence_dependency_ids=evidence_dependency_ids,
            consumed_dependency_ids=tuple(
                sorted({*direct_dependencies, *evidence_dependency_ids})
            ),
            control_dependency_ids=(),
            checksums={
                **{
                    path: tree_or_blob_digest(payload)
                    for path, (payload, _) in source_files.items()
                },
                **{
                    path: tree_or_blob_digest(payload)
                    for path, (payload, _) in evidence_files.items()
                },
                EXPERT_RELEASE_SOURCE_ARCHIVE: tree_or_blob_digest(source_archive),
                EXPERT_RELEASE_EVIDENCE_ARCHIVE: tree_or_blob_digest(evidence_archive),
            },
        )
        control_archive = self._archive(
            {
                EXPERT_RELEASE_MANIFEST_PATH: (
                    manifest.to_json_bytes(),
                    _REGULAR_MODE,
                )
            }
        )
        package = ExpertReleasePackage(
            manifest=manifest,
            evidence_manifest=evidence_manifest,
            matrix_summary=matrix_summary,
            source_files=source_files,
            evidence_files=evidence_files,
            source_archive=source_archive,
            evidence_archive=evidence_archive,
            control_archive=control_archive,
        )
        self.verify(package)
        if (
            self.candidate_store.read(candidate_id) != stored_candidate
            or self.validation_store.snapshot(candidate_id) != approved_snapshot
        ):
            raise ExpertReleaseAssemblyError(
                "release authority changed during deterministic assembly"
            )
        return package

    def verify(self, package: ExpertReleasePackage) -> None:
        if type(package) is not ExpertReleasePackage:
            raise ExpertReleaseAssemblyError(
                "release verification requires an exact package"
            )
        expected_checksums = {
            **{
                path: tree_or_blob_digest(payload)
                for path, (payload, _) in package.source_files.items()
            },
            **{
                path: tree_or_blob_digest(payload)
                for path, (payload, _) in package.evidence_files.items()
            },
            package.manifest.source_archive_ref: tree_or_blob_digest(
                package.source_archive
            ),
            package.manifest.evidence_archive_ref: tree_or_blob_digest(
                package.evidence_archive
            ),
        }
        source_tree_hash = source_tree_digest(
            {
                path: (tree_or_blob_digest(payload), mode, len(payload))
                for path, (payload, mode) in package.source_files.items()
            }
        )
        evidence_record_checksums = {
            path: tree_or_blob_digest(payload)
            for path, (payload, _) in package.evidence_files.items()
            if path != EXPERT_RELEASE_EVIDENCE_MANIFEST_PATH
        }
        evidence_records = self._verify_evidence_record_identities(
            package.evidence_files
        )
        observed_record_ids = tuple(sorted(evidence_records))
        evidence_dependency_ids = self._evidence_dependency_ids(package.evidence_files)
        manifest = package.manifest
        evidence = package.evidence_manifest
        summary = package.matrix_summary
        if (
            dict(manifest.checksums) != expected_checksums
            or manifest.candidate_tree_hash != source_tree_hash
            or package.evidence_files[EXPERT_RELEASE_EVIDENCE_MANIFEST_PATH][0]
            != evidence.to_json_bytes()
            or package.evidence_files[EXPERT_RELEASE_MATRIX_SUMMARY_PATH][0]
            != summary.to_json_bytes()
            or dict(evidence.record_checksums) != evidence_record_checksums
            or evidence.record_ids != observed_record_ids
            or not self._evidence_records_join_manifest(manifest, evidence_records)
            or manifest.evidence_dependency_ids != evidence_dependency_ids
            or set(manifest.consumed_dependency_ids)
            != {
                manifest.scope_contract_id,
                manifest.candidate_id,
                manifest.candidate_commit_record_id,
                manifest.candidate_tree_ref,
                manifest.candidate_derivation_ref,
                manifest.candidate_validation_context_ref,
                manifest.candidate_patch_ref,
                manifest.candidate_sanitation_report_id,
                *manifest.candidate_ancestor_ids,
                *manifest.candidate_source_dependency_ids,
                *manifest.candidate_consumed_expert_release_ids,
                manifest.repository_map_ref,
                *manifest.module_contract_refs,
                manifest.validation_attempt_id,
                manifest.approval_transition_id,
                manifest.approval_state_id,
                manifest.publication_eligibility_result_id,
                manifest.release_matrix_stage_result_id,
                manifest.release_matrix_report_id,
                manifest.promotion_decision_id,
                *manifest.approval_assertion_ids,
                manifest.validation_policy_id,
                manifest.evidence_manifest_ref,
                manifest.test_matrix_summary_ref,
                *manifest.evidence_dependency_ids,
                *(
                    (manifest.lineage.source_base_release_id,)
                    if manifest.lineage.source_base_release_id
                    else ()
                ),
            }
            or manifest.control_dependency_ids
            or (
                manifest.candidate_id,
                manifest.candidate_commit_record_id,
                manifest.candidate_tree_hash,
                manifest.validation_attempt_id,
                manifest.approval_transition_id,
                manifest.approval_state_id,
                manifest.publication_eligibility_result_id,
                manifest.release_matrix_stage_result_id,
                manifest.release_matrix_report_id,
                manifest.promotion_decision_id,
                manifest.evidence_manifest_ref,
                manifest.test_matrix_summary_ref,
            )
            != (
                evidence.candidate_id,
                evidence.candidate_commit_record_id,
                evidence.candidate_tree_hash,
                evidence.validation_attempt_id,
                evidence.approval_transition_id,
                evidence.approval_state_id,
                evidence.publication_eligibility_result_id,
                summary.release_matrix_stage_result_id,
                summary.release_matrix_report_id,
                summary.promotion_decision_id,
                evidence.evidence_manifest_id,
                summary.summary_id,
            )
            or package.source_archive != self._archive(package.source_files)
            or package.evidence_archive != self._archive(package.evidence_files)
            or package.control_archive
            != self._archive(
                {
                    EXPERT_RELEASE_MANIFEST_PATH: (
                        package.manifest.to_json_bytes(),
                        _REGULAR_MODE,
                    )
                }
            )
        ):
            raise ExpertReleaseAssemblyError(
                "expert release package differs from deterministic assembly"
            )

    @staticmethod
    def _verify_evidence_record_identities(
        evidence_files: Mapping[str, tuple[bytes, str]],
    ) -> dict[str, Mapping[str, object]]:
        records: dict[str, Mapping[str, object]] = {}
        for path, (payload, mode) in sorted(evidence_files.items()):
            if path in {
                EXPERT_RELEASE_EVIDENCE_MANIFEST_PATH,
                EXPERT_RELEASE_MATRIX_SUMMARY_PATH,
            }:
                continue
            prefix = f"{_RELEASE_RECORD_ROOT}/"
            if not path.startswith(prefix) or not path.endswith(".json"):
                raise ExpertReleaseAssemblyError(
                    "release evidence contains an undeclared record path"
                )
            namespace, digest_filename = path[len(prefix) :].split("/", 1)
            digest = digest_filename.removesuffix(".json")
            expected_id = f"{namespace}:sha256:{digest}"
            parsed = parse_json_bytes(payload)
            if not isinstance(parsed, Mapping) or mode != _REGULAR_MODE:
                raise ExpertReleaseAssemblyError(
                    "release evidence record payload is invalid"
                )
            identity_fields = tuple(
                key for key, value in parsed.items() if value == expected_id
            )
            if len(identity_fields) != 1:
                raise ExpertReleaseAssemblyError(
                    "release evidence record does not declare its path identity"
                )
            identity_field = identity_fields[0]
            content = dict(parsed)
            content.pop(identity_field)
            if (
                canonical_json_bytes(parsed) != payload
                or content_id(namespace, content) != expected_id
            ):
                raise ExpertReleaseAssemblyError(
                    "release evidence record content identity is invalid"
                )
            records[expected_id] = parsed
        return records

    @staticmethod
    def _evidence_records_join_manifest(
        manifest: ExpertBaseReleaseManifest,
        records: Mapping[str, Mapping[str, object]],
    ) -> bool:
        candidate = records.get(manifest.candidate_id)
        commit = records.get(manifest.candidate_commit_record_id)
        attempt = records.get(manifest.validation_attempt_id)
        transition = records.get(manifest.approval_transition_id)
        state = records.get(manifest.approval_state_id)
        publication = records.get(manifest.publication_eligibility_result_id)
        matrix = records.get(manifest.release_matrix_stage_result_id)
        if any(
            value is None
            for value in (
                candidate,
                commit,
                attempt,
                transition,
                state,
                publication,
                matrix,
            )
        ):
            return False
        accepted = state.get("accepted_stage_results")
        if not isinstance(accepted, list):
            return False
        safe_accepted_ids = {
            value.get("stage_result_record_id")
            for value in accepted
            if isinstance(value, Mapping)
            and isinstance(value.get("stage_result_record_id"), str)
            and not value["stage_result_record_id"].startswith(
                "expert-evaluator-result-record:sha256:"
            )
        }
        required_record_ids = {
            manifest.candidate_id,
            manifest.candidate_commit_record_id,
            manifest.candidate_tree_ref,
            manifest.candidate_derivation_ref,
            manifest.candidate_validation_context_ref,
            manifest.candidate_patch_ref,
            manifest.candidate_sanitation_report_id,
            manifest.repository_map_ref,
            *manifest.module_contract_refs,
            manifest.validation_attempt_id,
            manifest.approval_transition_id,
            manifest.approval_state_id,
            manifest.publication_eligibility_result_id,
            manifest.release_matrix_stage_result_id,
            *safe_accepted_ids,
        }
        promotion = publication.get("promotion_decision")
        report = matrix.get("release_matrix_report")
        return (
            set(records) == required_record_ids
            and candidate.get("candidate_tree_ref") == manifest.candidate_tree_ref
            and candidate.get("candidate_tree_hash") == manifest.candidate_tree_hash
            and candidate.get("derivation_ref") == manifest.candidate_derivation_ref
            and candidate.get("validation_context_ref")
            == manifest.candidate_validation_context_ref
            and candidate.get("patch_ref") == manifest.candidate_patch_ref
            and candidate.get("sanitation_report_id")
            == manifest.candidate_sanitation_report_id
            and candidate.get("proposed_repository_map_ref")
            == manifest.repository_map_ref
            and tuple(candidate.get("module_contract_refs", ()))
            == manifest.module_contract_refs
            and tuple(candidate.get("ancestor_candidate_ids", ()))
            == manifest.candidate_ancestor_ids
            and tuple(candidate.get("source_dependency_ids", ()))
            == manifest.candidate_source_dependency_ids
            and tuple(candidate.get("consumed_expert_release_ids", ()))
            == manifest.candidate_consumed_expert_release_ids
            and commit.get("candidate_id") == manifest.candidate_id
            and attempt.get("candidate_id") == manifest.candidate_id
            and attempt.get("candidate_tree_hash") == manifest.candidate_tree_hash
            and attempt.get("candidate_commit_record_id")
            == manifest.candidate_commit_record_id
            and attempt.get("validation_policy_id") == manifest.validation_policy_id
            and attempt.get("configuration_fingerprint")
            == manifest.configuration_fingerprint
            and transition.get("candidate_id") == manifest.candidate_id
            and transition.get("target_state_id") == manifest.approval_state_id
            and state.get("candidate_id") == manifest.candidate_id
            and state.get("candidate_tree_hash") == manifest.candidate_tree_hash
            and state.get("promotion_state") == "approved"
            and tuple(state.get("review_assertion_ids", ()))
            == manifest.approval_assertion_ids
            and publication.get("candidate_id") == manifest.candidate_id
            and publication.get("candidate_commit_record_id")
            == manifest.candidate_commit_record_id
            and isinstance(promotion, Mapping)
            and promotion.get("promotion_decision_id") == manifest.promotion_decision_id
            and promotion.get("release_matrix_stage_result_id")
            == manifest.release_matrix_stage_result_id
            and isinstance(report, Mapping)
            and report.get("release_matrix_report_id")
            == manifest.release_matrix_report_id
        )

    @staticmethod
    def _evidence_dependency_ids(
        evidence_files: Mapping[str, tuple[bytes, str]],
    ) -> tuple[str, ...]:
        dependencies: set[str] = set()

        def collect(value: object) -> None:
            if isinstance(value, str):
                if is_content_id(value):
                    dependencies.add(value)
                return
            if isinstance(value, Mapping):
                for key, nested in value.items():
                    collect(key)
                    collect(nested)
                return
            if isinstance(value, (list, tuple)):
                for nested in value:
                    collect(nested)

        for payload, _ in evidence_files.values():
            collect(parse_json_bytes(payload))
        return tuple(sorted(dependencies))

    def _archive(self, files: Mapping[str, tuple[bytes, str]]) -> bytes:
        return build_deterministic_tar_zst(
            files,
            compression_level=self.expert_settings.release_archive_compression_level,
            zstd_window_size_bytes=self.github_settings.zstd_window_size_bytes,
        )

    @staticmethod
    def _source_files(
        stored_candidate: StoredExpertCandidate,
    ) -> dict[str, tuple[bytes, str]]:
        closure = stored_candidate.closure
        descriptors = {
            descriptor.relative_path: descriptor
            for descriptor in closure.candidate_tree.files
        }
        reserved_paths = {
            EXPERT_RELEASE_MANIFEST_PATH,
            EXPERT_RELEASE_SOURCE_ARCHIVE,
            EXPERT_RELEASE_EVIDENCE_ARCHIVE,
            EXPERT_RELEASE_CONTROL_ARCHIVE,
        }
        for path in closure.candidate_contents:
            if path in reserved_paths or path.startswith(
                f"{EXPERT_RELEASE_EVIDENCE_ROOT}/"
            ):
                raise ExpertReleaseAssemblyError(
                    "candidate source occupies a release-reserved path"
                )
        files = {
            path: (payload, descriptors[path].mode)
            for path, payload in closure.candidate_contents.items()
        }
        if set(files) != set(descriptors):
            raise ExpertReleaseAssemblyError(
                "candidate source differs from its approved descriptor closure"
            )
        return dict(sorted(files.items()))

    @staticmethod
    def _validate_approved_snapshot(
        *,
        stored_candidate: StoredExpertCandidate,
        snapshot: ExpertValidationSnapshot,
    ) -> tuple[
        ExpertPublicationEligibilityStageResultRecord,
        ExpertReleaseMatrixStageResultRecord,
        ExpertAutomatedReviewStageResultRecord,
    ]:
        candidate = stored_candidate.closure.manifest
        attempt = snapshot.latest_attempt
        publication_results = tuple(
            result
            for result in snapshot.accepted_stage_results
            if type(result) is ExpertPublicationEligibilityStageResultRecord
        )
        matrix_results = tuple(
            result
            for result in snapshot.accepted_stage_results
            if type(result) is ExpertReleaseMatrixStageResultRecord
        )
        review_results = tuple(
            result
            for result in snapshot.accepted_stage_results
            if type(result) is ExpertAutomatedReviewStageResultRecord
        )
        if (
            type(snapshot.transition) is not ExpertValidationTransition
            or attempt is None
            or type(attempt) is not ExpertValidationAttempt
            or snapshot.state.promotion_state is not ExpertPromotionState.APPROVED
            or snapshot.state.next_stage is not None
            or len(publication_results) != 1
            or len(matrix_results) != 1
            or len(review_results) != 1
            or snapshot.accepted_stage_results[-1] != publication_results[0]
        ):
            raise ExpertReleaseAssemblyError(
                "release assembly requires one complete terminal approval cascade"
            )
        publication_result = publication_results[0]
        matrix_result = matrix_results[0]
        review_result = review_results[0]
        decision = publication_result.promotion_decision
        if (
            decision.outcome is not ExpertReleaseMatrixDecisionOutcome.APPROVED
            or publication_result.release_use_decision is None
            or publication_result.release_use_decision.outcome
            is not ExpertCandidateReleaseUseOutcome.CLEARED
            or publication_result.publication_authority_fence is None
            or snapshot.transition.target_state_id != snapshot.state.validation_state_id
            or snapshot.transition.transition_stage_result_record_id
            != publication_result.stage_result_record_id
            or snapshot.transition.accepted_stage_result_record_ids
            != tuple(
                ExpertReleaseAssembler._record_id(result)
                for result in snapshot.accepted_stage_results
            )
            or snapshot.state.candidate_id != candidate.candidate_id
            or snapshot.state.candidate_tree_hash != candidate.candidate_tree_hash
            or attempt.candidate_id != candidate.candidate_id
            or attempt.candidate_tree_hash != candidate.candidate_tree_hash
            or attempt.candidate_commit_record_id
            != stored_candidate.commit_record.commit_record_id
            or attempt.scope_contract_id != candidate.scope_contract_id
            or attempt.source_base_release_id != candidate.source_base_release_id
            or publication_result.candidate_commit_record_id
            != stored_candidate.commit_record.commit_record_id
        ):
            raise ExpertReleaseAssemblyError(
                "terminal approval differs from its exact candidate authority"
            )
        if (
            publication_result.promotion_decision.release_matrix_stage_result_id
            != matrix_result.stage_result_record_id
            or matrix_result.release_matrix_report.candidate_commit_record_id
            != stored_candidate.commit_record.commit_record_id
            or tuple(sorted(review_result.assertion_ids))
            != snapshot.state.review_assertion_ids
            or not set(
                stored_candidate_admission_dependency_ids(stored_candidate)
            ).issubset(publication_result.exact_dependency_ids)
        ):
            raise ExpertReleaseAssemblyError(
                "terminal approval omits matrix, review, or admission authority"
            )
        return publication_result, matrix_result, review_result

    @staticmethod
    def _matrix_summary(
        matrix_result: ExpertReleaseMatrixStageResultRecord,
        publication_result: ExpertPublicationEligibilityStageResultRecord,
    ) -> ExpertReleaseMatrixSummary:
        report = matrix_result.release_matrix_report
        plan = report.evaluation_plan
        return ExpertReleaseMatrixSummary.mint(
            release_matrix_stage_result_id=matrix_result.stage_result_record_id,
            release_matrix_report_id=report.release_matrix_report_id,
            promotion_decision_id=(
                publication_result.promotion_decision.promotion_decision_id
            ),
            mode=report.mode,
            outcome=publication_result.promotion_decision.outcome,
            evaluation_cell_count=len(plan.evaluation_cells),
            provenance_count=len(plan.provenance_bindings),
            task_adapter_count=len(plan.adapter_authorities),
        )

    @staticmethod
    def _evidence_records(
        *,
        stored_candidate: StoredExpertCandidate,
        snapshot: ExpertValidationSnapshot,
    ) -> tuple[StrictContract, ...]:
        closure = stored_candidate.closure
        attempt = snapshot.latest_attempt
        if attempt is None:
            raise ExpertReleaseAssemblyError(
                "release evidence requires a validation attempt"
            )
        records: tuple[StrictContract, ...] = (
            closure.manifest,
            stored_candidate.commit_record,
            closure.validation_context,
            closure.patch,
            closure.candidate_tree,
            closure.repository_map,
            *closure.module_contracts,
            closure.sanitation_report,
            closure.derivation.record,
            attempt,
            snapshot.transition,
            snapshot.state,
            *(
                result
                for result in snapshot.accepted_stage_results
                if type(result) is not ExpertEvaluatorResultRecord
            ),
        )
        ids = tuple(ExpertReleaseAssembler._record_id(record) for record in records)
        if len(ids) != len(set(ids)):
            raise ExpertReleaseAssemblyError(
                "release evidence repeats one content-addressed record"
            )
        return tuple(sorted(records, key=ExpertReleaseAssembler._record_id))

    @staticmethod
    def _record_id(record: StrictContract) -> str:
        identity_field = record.IDENTITY_FIELD
        if identity_field is None:
            raise ExpertReleaseAssemblyError(
                "release evidence record is not content addressed"
            )
        value = getattr(record, identity_field)
        if not isinstance(value, str):
            raise ExpertReleaseAssemblyError("release evidence identity is invalid")
        return value

    @staticmethod
    def _record_path(record: StrictContract) -> str:
        record_id = ExpertReleaseAssembler._record_id(record)
        namespace, digest = record_id.split(":sha256:", 1)
        return f"{_RELEASE_RECORD_ROOT}/{namespace}/{digest}.json"


__all__ = [
    "EXPERT_RELEASE_CONTROL_ARCHIVE",
    "EXPERT_RELEASE_EVIDENCE_ARCHIVE",
    "EXPERT_RELEASE_EVIDENCE_MANIFEST_PATH",
    "EXPERT_RELEASE_MANIFEST_PATH",
    "EXPERT_RELEASE_MATRIX_SUMMARY_PATH",
    "EXPERT_RELEASE_SOURCE_ARCHIVE",
    "ExpertReleaseAssembler",
    "ExpertReleaseAssemblyError",
    "ExpertReleasePackage",
]
