"""Private, create-only expert candidate package store."""

from __future__ import annotations

import ctypes
import errno
import fcntl
import os
import re
import stat
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Mapping

from kapso.cross_run.agent_artifacts import (
    CodingAgentWorkspaceAccess,
    coding_agent_artifact_filenames,
)
from kapso.cross_run.canonical import (
    parse_json_bytes,
    require_content_id,
    tree_or_blob_digest,
)
from kapso.cross_run.capture.safety import (
    remove_restricted_directory,
    restricted_directory_identity,
)
from kapso.cross_run.contracts import (
    CodingAgentWorkspaceDelta,
    EXPERT_CANDIDATE_COMMIT_PATH,
    ExpertCandidateCommitRecord,
    ExpertCandidateDerivationKind,
    ExpertCandidateManifest,
    ExpertCandidateOperationRecord,
    ExpertCandidatePatch,
    ExpertCandidateSanitationReport,
    ExpertModuleContract,
    ExpertRepositoryMap,
    ExpertSourceTreeManifest,
    SourceFileDescriptor,
    StrictContract,
)
from kapso.cross_run.expert.candidates import (
    ExpertCandidateClosure,
    ExpertCandidateValidator,
    composition_source_candidate_closure,
)
from kapso.cross_run.expert.candidate_context import (
    ExpertCandidateValidationContext,
)
from kapso.cross_run.expert.candidate_package import (
    AGENT_ANCESTORS_PACKAGE_PATH,
    AGENT_ARTIFACT_PACKAGE_ROOT,
    AGENT_OPERATION_PACKAGE_PATH,
    AGENT_TRIGGER_DECISION_PACKAGE_PATH,
    AGENT_TRIGGER_PACKET_PACKAGE_PATH,
    AGENT_WORKSPACE_DELTA_PACKAGE_PATH,
    PARENT_FILES_PACKAGE_PATH,
    SANITATION_REPORT_PACKAGE_PATH,
    contract_tuple_package_bytes,
    direct_agent_candidate_package_files,
)
from kapso.cross_run.expert.candidate_derivations import (
    AGENT_DERIVATION_RECORD_PACKAGE_PATH,
    CANDIDATE_MANIFEST_PACKAGE_PATH,
    CANDIDATE_MODULE_PACKAGE_ROOT,
    CANDIDATE_PATCH_PACKAGE_PATH,
    CANDIDATE_REPOSITORY_MAP_PACKAGE_PATH,
    CANDIDATE_SOURCE_PACKAGE_ROOT,
    CANDIDATE_SOURCE_TREE_PACKAGE_PATH,
    CANDIDATE_VALIDATION_CONTEXT_PACKAGE_PATH,
    COMPOSITION_DERIVATION_RECORD_PACKAGE_PATH,
    ExpertAgentProposalDerivation,
    ExpertAgentProposalDerivationRecord,
    ExpertCompositionSourceProvenance,
    ExpertDeterministicCompositionDerivation,
    ExpertDeterministicCompositionDerivationRecord,
)
from kapso.cross_run.expert.composition import ExpertCompositionReductionSource
from kapso.cross_run.expert.composition_admission_contracts import (
    ExpertCompositionAdmissionFence,
    validate_expert_composition_admission_fence,
)
from kapso.cross_run.expert.composition_admission_authority import (
    ExpertCompositionAdmissionAuthority,
    ExpertCompositionApprovalLease,
)
from kapso.cross_run.expert.composition_contracts import (
    ExpertCompositionMaterialization,
    ExpertCompositionSourceReference,
)
from kapso.cross_run.expert.proposal_contract import ExpertCandidateAncestorInput
from kapso.cross_run.expert.triggers import (
    ExpertEvolutionTriggerDecision,
    ExpertTriggerEvidencePacket,
)

_COMMIT_PATH = EXPERT_CANDIDATE_COMMIT_PATH
_MANIFEST_PATH = CANDIDATE_MANIFEST_PACKAGE_PATH
_VALIDATION_CONTEXT_PATH = CANDIDATE_VALIDATION_CONTEXT_PACKAGE_PATH
_PATCH_PATH = CANDIDATE_PATCH_PACKAGE_PATH
_SOURCE_TREE_PATH = CANDIDATE_SOURCE_TREE_PACKAGE_PATH
_PARENT_FILES_PATH = PARENT_FILES_PACKAGE_PATH
_REPOSITORY_MAP_PATH = CANDIDATE_REPOSITORY_MAP_PACKAGE_PATH
_SANITATION_PATH = SANITATION_REPORT_PACKAGE_PATH
_MODULE_ROOT = CANDIDATE_MODULE_PACKAGE_ROOT
_SOURCE_ROOT = CANDIDATE_SOURCE_PACKAGE_ROOT
_AGENT_DERIVATION_ROOT = "derivations/agent"
_AGENT_DERIVATION_RECORD_PATH = AGENT_DERIVATION_RECORD_PACKAGE_PATH
_TRIGGER_PACKET_PATH = AGENT_TRIGGER_PACKET_PACKAGE_PATH
_TRIGGER_DECISION_PATH = AGENT_TRIGGER_DECISION_PACKAGE_PATH
_OPERATION_PATH = AGENT_OPERATION_PACKAGE_PATH
_WORKSPACE_DELTA_PATH = AGENT_WORKSPACE_DELTA_PACKAGE_PATH
_ANCESTORS_PATH = AGENT_ANCESTORS_PACKAGE_PATH
_AGENT_ARTIFACT_ROOT = AGENT_ARTIFACT_PACKAGE_ROOT
_COMPOSITION_DERIVATION_ROOT = "derivations/composition"
_COMPOSITION_DERIVATION_RECORD_PATH = COMPOSITION_DERIVATION_RECORD_PACKAGE_PATH
_COMPOSITION_MATERIALIZATION_PATH = (
    f"{_COMPOSITION_DERIVATION_ROOT}/materialization.json"
)
_COMPOSITION_PARENT_SOURCE_ROOT = f"{_COMPOSITION_DERIVATION_ROOT}/parent-source"
_COMPOSITION_SOURCE_PROVENANCE_ROOT = (
    f"{_COMPOSITION_DERIVATION_ROOT}/source-provenance"
)
_COMPOSITION_ADMISSION_PATH = "ADMISSION.json"
_SEALED_COMPOSITION_ADMISSION = object()
_RENAME_NOREPLACE = 1
_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_STAGING_PATTERN = re.compile(r"^\.candidate-[A-Za-z0-9_-]+$")


class ExpertCandidateStoreError(ValueError):
    """Candidate package storage is unsafe, corrupt, or conflicting."""


@dataclass(frozen=True)
class StoredExpertCandidate:
    root: Path
    closure: ExpertCandidateClosure
    commit_record: ExpertCandidateCommitRecord
    composition_admission_fence: ExpertCompositionAdmissionFence | None = None


def stored_candidate_admission_dependency_ids(
    stored_candidate: StoredExpertCandidate,
) -> tuple[str, ...]:
    """Project durable admission identity that every validation attempt must pin."""

    if type(stored_candidate) is not StoredExpertCandidate:
        raise ExpertCandidateStoreError(
            "candidate admission dependency projection requires one stored candidate"
        )
    fence = stored_candidate.composition_admission_fence
    if fence is not None and type(fence) is not ExpertCompositionAdmissionFence:
        raise ExpertCandidateStoreError(
            "stored candidate admission authority uses another type"
        )
    if fence is None:
        return ()
    return tuple(
        sorted(
            {
                fence.admission_fence_id,
                fence.security_denylist_observation.observation_id,
                *fence.exact_dependency_ids,
                *fence.security_subject_ids,
            }
        )
    )


class ExpertCandidateStore:
    """Atomically seal and reopen complete candidate closures."""

    def __init__(
        self,
        root: Path,
        state_root: Path,
        validator: ExpertCandidateValidator,
    ):
        self._validate_state_root(state_root)
        if (
            not root.is_absolute()
            or root != Path(os.path.abspath(root))
            or root.parent != state_root
            or root.name in {"", ".", ".."}
        ):
            raise ExpertCandidateStoreError(
                "candidate store must be a direct normalized child of its state root"
            )
        self.root = root
        self.state_root = state_root
        self.validator = validator
        self._composition_admission_authority = None
        self.object_root = root / "objects"
        self.staging_root = root / "staging"
        initialization_lock = state_root / f".{root.name}.initialization.lock"
        with _CandidateStoreLock(
            initialization_lock,
            exclusive=True,
            create=True,
        ):
            self._prepare_layout()
        with self._exclusive_lock():
            self._recover_staging()

    def persist(self, closure: ExpertCandidateClosure) -> StoredExpertCandidate:
        if type(closure.derivation) is ExpertDeterministicCompositionDerivation:
            raise ExpertCandidateStoreError(
                "composition persistence requires sealed admission authority"
            )
        snapshot, package_files, commit_record = self._prepare_candidate(closure)
        return self._persist_prepared(
            snapshot=snapshot,
            package_files=package_files,
            commit_record=commit_record,
            composition_admission_fence=None,
        )

    def preview_composition_commit(
        self,
        closure: ExpertCandidateClosure,
    ) -> ExpertCandidateCommitRecord:
        """Compute the commit that an admitted composition would persist."""

        if type(closure.derivation) is not ExpertDeterministicCompositionDerivation:
            raise ExpertCandidateStoreError(
                "composition commit preview requires a composition candidate"
            )
        _, _, commit_record = self._prepare_candidate(closure)
        return commit_record

    def _bind_composition_admission_authority(
        self,
        authority: ExpertCompositionAdmissionAuthority,
    ) -> None:
        if type(authority) is not ExpertCompositionAdmissionAuthority:
            raise ExpertCandidateStoreError(
                "candidate store requires its exact composition admission authority"
            )
        authority._require_bound(candidate_store=self)
        with self._exclusive_lock():
            if (
                self._composition_admission_authority is not None
                and self._composition_admission_authority is not authority
            ):
                raise ExpertCandidateStoreError(
                    "candidate store already has another composition admission authority"
                )
            self._composition_admission_authority = authority

    def _seal_composition_admission(
        self,
        *,
        authority: ExpertCompositionAdmissionAuthority,
        approval_lease: ExpertCompositionApprovalLease,
        closure: ExpertCandidateClosure,
        freshness_context: object,
    ) -> _SealedExpertCompositionAdmission:
        if (
            type(authority) is not ExpertCompositionAdmissionAuthority
            or authority is not self._composition_admission_authority
        ):
            raise ExpertCandidateStoreError(
                "composition admission was prepared by a foreign authority"
            )
        approved_sources = freshness_context.approved_sources
        authority._require_approval_lease(
            candidate_store=self,
            approval_lease=approval_lease,
            approved_sources=approved_sources,
        )
        snapshot, package_files, commit_record = self._prepare_candidate(closure)
        if type(snapshot.derivation) is not ExpertDeterministicCompositionDerivation:
            raise ExpertCandidateStoreError(
                "composition admission cannot seal another derivation"
            )
        return _SealedExpertCompositionAdmission(
            seal=_SEALED_COMPOSITION_ADMISSION,
            store=self,
            authority=authority,
            approval_lease=approval_lease,
            snapshot=snapshot,
            package_files=package_files,
            commit_record=commit_record,
            freshness_context=freshness_context,
        )

    def _commit_composition_admission(
        self,
        *,
        authority: ExpertCompositionAdmissionAuthority,
        admission: _SealedExpertCompositionAdmission,
    ) -> StoredExpertCandidate:
        if (
            type(authority) is not ExpertCompositionAdmissionAuthority
            or authority is not self._composition_admission_authority
            or type(admission) is not _SealedExpertCompositionAdmission
        ):
            raise ExpertCandidateStoreError(
                "composition admission commit uses a foreign authority"
            )
        with self._exclusive_lock():
            self._recover_staging()
            (
                snapshot,
                package_files,
                commit_record,
                freshness_context,
                approval_lease,
            ) = admission._consume(
                store=self,
                authority=authority,
            )
            authority._require_approval_lease(
                candidate_store=self,
                approval_lease=approval_lease,
                approved_sources=freshness_context.approved_sources,
            )
            (
                validated_snapshot,
                validated_package_files,
                validated_commit_record,
            ) = self._prepare_candidate(snapshot)
            if (
                validated_snapshot != snapshot
                or validated_package_files != package_files
                or validated_commit_record != commit_record
            ):
                raise ExpertCandidateStoreError(
                    "composition admission changed before locked persistence"
                )
            self._validate_composition_sources_unlocked(snapshot)
            fence = authority._finalize_under_store_lock(
                candidate_store=self,
                freshness_context=freshness_context,
                closure=snapshot,
                commit_record=commit_record,
            )
            validate_expert_composition_admission_fence(
                fence=fence,
                closure=snapshot,
                commit_record=commit_record,
            )
            return self._persist_prepared_unlocked(
                snapshot=snapshot,
                package_files=package_files,
                commit_record=commit_record,
                composition_admission_fence=fence,
            )

    def _prepare_candidate(
        self,
        closure: ExpertCandidateClosure,
    ) -> tuple[
        ExpertCandidateClosure,
        dict[str, bytes],
        ExpertCandidateCommitRecord,
    ]:
        snapshot = self._snapshot_closure(closure)
        self.validator.validate(snapshot)
        package_files = self._package_files(snapshot)
        commit_record = ExpertCandidateCommitRecord.mint(
            candidate_id=snapshot.manifest.candidate_id,
            file_checksums={
                path: tree_or_blob_digest(payload)
                for path, payload in sorted(package_files.items())
            },
        )
        return snapshot, package_files, commit_record

    def _persist_prepared(
        self,
        *,
        snapshot: ExpertCandidateClosure,
        package_files: dict[str, bytes],
        commit_record: ExpertCandidateCommitRecord,
        composition_admission_fence: ExpertCompositionAdmissionFence | None,
    ) -> StoredExpertCandidate:
        if (
            type(snapshot.derivation) is ExpertDeterministicCompositionDerivation
            or composition_admission_fence is not None
        ):
            raise ExpertCandidateStoreError(
                "generic candidate persistence cannot store composition authority"
            )
        with self._exclusive_lock():
            self._recover_staging()
            return self._persist_prepared_unlocked(
                snapshot=snapshot,
                package_files=package_files,
                commit_record=commit_record,
                composition_admission_fence=composition_admission_fence,
            )

    def _persist_prepared_unlocked(
        self,
        *,
        snapshot: ExpertCandidateClosure,
        package_files: dict[str, bytes],
        commit_record: ExpertCandidateCommitRecord,
        composition_admission_fence: ExpertCompositionAdmissionFence | None,
    ) -> StoredExpertCandidate:
        destination = self._candidate_path(snapshot.manifest.candidate_id)
        if os.path.lexists(destination):
            stored = self._read_unlocked(snapshot.manifest.candidate_id)
            if stored.closure != snapshot or stored.commit_record != commit_record:
                raise ExpertCandidateStoreError(
                    "candidate identity conflicts with persisted closure"
                )
            return stored
        with tempfile.TemporaryDirectory(
            prefix=".candidate-",
            dir=self.staging_root,
        ) as staging_name:
            staging = Path(staging_name)
            staging.chmod(0o700)
            for relative_path, payload in package_files.items():
                self._write_private_file(staging, relative_path, payload)
            self._write_private_file(
                staging,
                _COMMIT_PATH,
                commit_record.to_json_bytes(),
            )
            if composition_admission_fence is not None:
                self._write_private_file(
                    staging,
                    _COMPOSITION_ADMISSION_PATH,
                    composition_admission_fence.to_json_bytes(),
                )
            self._fsync_tree(staging)
            staged = self._read_package(
                staging,
                snapshot.manifest.candidate_id,
            )
            if (
                staged.closure != snapshot
                or staged.commit_record != commit_record
                or staged.composition_admission_fence != composition_admission_fence
            ):
                raise ExpertCandidateStoreError(
                    "staged candidate differs from its validated snapshot"
                )
            self._rename_directory_no_replace(staging, destination)
            self._fsync_directory(self.staging_root)
            self._fsync_directory(self.object_root)
        return self._read_unlocked(snapshot.manifest.candidate_id)

    def _validate_composition_sources_unlocked(
        self,
        closure: ExpertCandidateClosure,
    ) -> None:
        derivation = closure.derivation
        if type(derivation) is not ExpertDeterministicCompositionDerivation:
            raise ExpertCandidateStoreError(
                "composition source validation requires its exact derivation"
            )
        for provenance in derivation.source_provenance:
            stored_source = self._read_unlocked(provenance.candidate_id)
            expected_source = composition_source_candidate_closure(provenance)
            if (
                stored_source.closure != expected_source
                or stored_source.commit_record != provenance.candidate_commit_record
                or stored_source.composition_admission_fence is not None
            ):
                raise ExpertCandidateStoreError(
                    "composition source changed before locked persistence"
                )

    @staticmethod
    def _snapshot_closure(
        closure: ExpertCandidateClosure,
    ) -> ExpertCandidateClosure:
        derivation = closure.derivation
        if type(derivation) is ExpertAgentProposalDerivation:
            snapshot_derivation = replace(
                derivation,
                operation_artifacts=dict(derivation.operation_artifacts),
            )
        elif type(derivation) is ExpertDeterministicCompositionDerivation:
            snapshot_derivation = replace(
                derivation,
                parent_contents=dict(derivation.parent_contents),
                source_provenance=tuple(
                    replace(
                        provenance,
                        reduction_source=replace(
                            provenance.reduction_source,
                            candidate_contents=dict(
                                provenance.reduction_source.candidate_contents
                            ),
                        ),
                        agent_derivation=replace(
                            provenance.agent_derivation,
                            operation_artifacts=dict(
                                provenance.agent_derivation.operation_artifacts
                            ),
                        ),
                    )
                    for provenance in derivation.source_provenance
                ),
            )
        else:
            raise ExpertCandidateStoreError(
                "candidate store does not recognize the derivation closure"
            )
        return replace(
            closure,
            candidate_contents=dict(closure.candidate_contents),
            derivation=snapshot_derivation,
        )

    def read(self, candidate_id: str) -> StoredExpertCandidate:
        require_content_id(candidate_id, "candidate_id")
        with self._shared_lock():
            return self._read_unlocked(candidate_id)

    def _read_unlocked(self, candidate_id: str) -> StoredExpertCandidate:
        return self._read_package(self._candidate_path(candidate_id), candidate_id)

    def _read_package(
        self,
        candidate_root: Path,
        candidate_id: str,
    ) -> StoredExpertCandidate:
        files = self._read_private_tree(candidate_root)
        commit_payload = files.get(_COMMIT_PATH)
        composition_admission_payload = files.get(_COMPOSITION_ADMISSION_PATH)
        if commit_payload is None:
            raise ExpertCandidateStoreError("candidate package is not committed")
        commit = ExpertCandidateCommitRecord.from_json_bytes(commit_payload)
        if commit_payload != commit.to_json_bytes():
            raise ExpertCandidateStoreError("candidate commit record is not canonical")
        payloads = {
            path: payload
            for path, payload in files.items()
            if path not in {_COMMIT_PATH, _COMPOSITION_ADMISSION_PATH}
        }
        if commit.candidate_id != candidate_id or set(payloads) != set(
            commit.file_checksums
        ):
            raise ExpertCandidateStoreError(
                "candidate package differs from its commit closure"
            )
        for path, payload in payloads.items():
            if tree_or_blob_digest(payload) != commit.file_checksums[path]:
                raise ExpertCandidateStoreError(
                    f"candidate package checksum differs: {path}"
                )
        closure = self._parse_closure(payloads)
        if closure.manifest.candidate_id != candidate_id:
            raise ExpertCandidateStoreError(
                "candidate directory names another manifest"
            )
        self.validator.validate_persisted(closure)
        composition_admission_fence = None
        if (
            closure.manifest.derivation_kind
            is ExpertCandidateDerivationKind.DETERMINISTIC_COMPOSITION
        ):
            if composition_admission_payload is None:
                raise ExpertCandidateStoreError(
                    "composition candidate lacks its admission fence"
                )
            composition_admission_fence = (
                ExpertCompositionAdmissionFence.from_json_bytes(
                    composition_admission_payload
                )
            )
            if (
                composition_admission_payload
                != composition_admission_fence.to_json_bytes()
            ):
                raise ExpertCandidateStoreError(
                    "composition admission fence is not canonical"
                )
            validate_expert_composition_admission_fence(
                fence=composition_admission_fence,
                closure=closure,
                commit_record=commit,
            )
        elif composition_admission_payload is not None:
            raise ExpertCandidateStoreError(
                "agent candidate cannot contain composition admission authority"
            )
        return StoredExpertCandidate(
            root=candidate_root,
            closure=closure,
            commit_record=commit,
            composition_admission_fence=composition_admission_fence,
        )

    @staticmethod
    def _package_files(closure: ExpertCandidateClosure) -> dict[str, bytes]:
        if type(closure.derivation) is ExpertAgentProposalDerivation:
            return direct_agent_candidate_package_files(
                manifest=closure.manifest,
                validation_context=closure.validation_context,
                patch=closure.patch,
                candidate_tree=closure.candidate_tree,
                parent_files=closure.parent_files,
                repository_map=closure.repository_map,
                module_contracts=closure.module_contracts,
                derivation=closure.derivation,
                sanitation_report=closure.sanitation_report,
                candidate_contents=closure.candidate_contents,
            )
        files = {
            _MANIFEST_PATH: closure.manifest.to_json_bytes(),
            _VALIDATION_CONTEXT_PATH: closure.validation_context.to_json_bytes(),
            _PATCH_PATH: closure.patch.to_json_bytes(),
            _SOURCE_TREE_PATH: closure.candidate_tree.to_json_bytes(),
            _PARENT_FILES_PATH: contract_tuple_package_bytes(closure.parent_files),
            _REPOSITORY_MAP_PATH: closure.repository_map.to_json_bytes(),
            _SANITATION_PATH: closure.sanitation_report.to_json_bytes(),
        }
        for module in closure.module_contracts:
            digest = module.module_contract_id.rsplit(":", 1)[1]
            files[f"{_MODULE_ROOT}/{digest}.json"] = module.to_json_bytes()
        for relative_path, payload in closure.candidate_contents.items():
            files[f"{_SOURCE_ROOT}/{relative_path}"] = payload
        if type(closure.derivation) is ExpertDeterministicCompositionDerivation:
            files.update(ExpertCandidateStore._composition_derivation_files(closure))
        else:
            raise ExpertCandidateStoreError(
                "candidate package uses an unknown derivation closure"
            )
        return dict(sorted(files.items()))

    @staticmethod
    def _composition_derivation_files(
        closure: ExpertCandidateClosure,
    ) -> dict[str, bytes]:
        derivation = closure.derivation
        if type(derivation) is not ExpertDeterministicCompositionDerivation:
            raise ExpertCandidateStoreError(
                "composition package requires a composition derivation"
            )
        files = {
            _COMPOSITION_DERIVATION_RECORD_PATH: derivation.record.to_json_bytes(),
            _COMPOSITION_MATERIALIZATION_PATH: (
                derivation.materialization.to_json_bytes()
            ),
        }
        for path, payload in derivation.parent_contents.items():
            files[f"{_COMPOSITION_PARENT_SOURCE_ROOT}/{path}"] = payload
        for provenance in derivation.source_provenance:
            root = ExpertCandidateStore._composition_source_root(
                provenance.candidate_id
            )
            files[f"{root}/commit.json"] = (
                provenance.candidate_commit_record.to_json_bytes()
            )
            source_closure = composition_source_candidate_closure(provenance)
            for path, payload in ExpertCandidateStore._package_files(
                source_closure
            ).items():
                files[f"{root}/package/{path}"] = payload
        return files

    @staticmethod
    def _parse_closure(payloads: Mapping[str, bytes]) -> ExpertCandidateClosure:
        manifest = ExpertCandidateManifest.from_json_bytes(payloads[_MANIFEST_PATH])
        tree = ExpertSourceTreeManifest.from_json_bytes(payloads[_SOURCE_TREE_PATH])
        modules = tuple(
            sorted(
                (
                    ExpertModuleContract.from_json_bytes(
                        payloads[f"{_MODULE_ROOT}/{module_ref.rsplit(':', 1)[1]}.json"]
                    )
                    for module_ref in manifest.module_contract_refs
                ),
                key=lambda module: module.module_id,
            )
        )
        candidate_contents = {
            file.relative_path: payloads[f"{_SOURCE_ROOT}/{file.relative_path}"]
            for file in tree.files
        }
        parent_files = ExpertCandidateStore._parse_contract_tuple(
            payloads[_PARENT_FILES_PATH],
            SourceFileDescriptor,
            "parent files",
        )
        closure = ExpertCandidateClosure(
            manifest=manifest,
            validation_context=ExpertCandidateValidationContext.from_json_bytes(
                payloads[_VALIDATION_CONTEXT_PATH]
            ),
            patch=ExpertCandidatePatch.from_json_bytes(payloads[_PATCH_PATH]),
            candidate_tree=tree,
            parent_files=parent_files,
            repository_map=ExpertRepositoryMap.from_json_bytes(
                payloads[_REPOSITORY_MAP_PATH]
            ),
            module_contracts=modules,
            derivation=ExpertCandidateStore._parse_derivation(manifest, payloads),
            sanitation_report=ExpertCandidateSanitationReport.from_json_bytes(
                payloads[_SANITATION_PATH]
            ),
            candidate_contents=candidate_contents,
        )
        expected_payloads = ExpertCandidateStore._package_files(closure)
        if dict(payloads) != expected_payloads:
            raise ExpertCandidateStoreError(
                "candidate package differs from its canonical closure"
            )
        return closure

    @staticmethod
    def _parse_derivation(
        manifest: ExpertCandidateManifest,
        payloads: Mapping[str, bytes],
    ):
        if manifest.derivation_kind is ExpertCandidateDerivationKind.AGENT_PROPOSAL:
            ancestors = ExpertCandidateStore._parse_contract_tuple(
                payloads[_ANCESTORS_PATH],
                ExpertCandidateAncestorInput,
                "ancestor inputs",
            )
            return ExpertAgentProposalDerivation(
                record=ExpertAgentProposalDerivationRecord.from_json_bytes(
                    payloads[_AGENT_DERIVATION_RECORD_PATH]
                ),
                trigger_packet=ExpertTriggerEvidencePacket.from_json_bytes(
                    payloads[_TRIGGER_PACKET_PATH]
                ),
                trigger_decision=ExpertEvolutionTriggerDecision.from_json_bytes(
                    payloads[_TRIGGER_DECISION_PATH]
                ),
                operation=ExpertCandidateOperationRecord.from_json_bytes(
                    payloads[_OPERATION_PATH]
                ),
                workspace_delta=CodingAgentWorkspaceDelta.from_json_bytes(
                    payloads[_WORKSPACE_DELTA_PATH]
                ),
                operation_artifacts={
                    name: payloads[f"{_AGENT_ARTIFACT_ROOT}/{name}"]
                    for name in coding_agent_artifact_filenames(
                        CodingAgentWorkspaceAccess.EDIT_WORKSPACE
                    )
                },
                ancestor_inputs=ancestors,
            )
        if manifest.derivation_kind is (
            ExpertCandidateDerivationKind.DETERMINISTIC_COMPOSITION
        ):
            return ExpertCandidateStore._parse_composition_derivation(payloads)
        raise ExpertCandidateStoreError(
            "candidate store does not recognize the derivation package"
        )

    @staticmethod
    def _parse_composition_derivation(
        payloads: Mapping[str, bytes],
    ) -> ExpertDeterministicCompositionDerivation:
        materialization = ExpertCompositionMaterialization.from_json_bytes(
            payloads[_COMPOSITION_MATERIALIZATION_PATH]
        )
        plan = materialization.composition_assessment.composition_plan
        provenance = tuple(
            ExpertCandidateStore._parse_composition_source_provenance(
                source_reference,
                payloads,
            )
            for source_reference in plan.sources
        )
        parent_contents = {
            descriptor.relative_path: payloads[
                f"{_COMPOSITION_PARENT_SOURCE_ROOT}/{descriptor.relative_path}"
            ]
            for descriptor in materialization.parent_tree.files
        }
        return ExpertDeterministicCompositionDerivation(
            record=ExpertDeterministicCompositionDerivationRecord.from_json_bytes(
                payloads[_COMPOSITION_DERIVATION_RECORD_PATH]
            ),
            materialization=materialization,
            source_provenance=provenance,
            parent_contents=parent_contents,
        )

    @staticmethod
    def _parse_composition_source_provenance(
        source_reference: ExpertCompositionSourceReference,
        payloads: Mapping[str, bytes],
    ) -> ExpertCompositionSourceProvenance:
        root = ExpertCandidateStore._composition_source_root(
            source_reference.candidate_id
        )
        commit = ExpertCandidateCommitRecord.from_json_bytes(
            payloads[f"{root}/commit.json"]
        )
        source_payloads = {
            path: payloads[f"{root}/package/{path}"] for path in commit.file_checksums
        }
        if any(
            tree_or_blob_digest(payload) != commit.file_checksums[path]
            for path, payload in source_payloads.items()
        ):
            raise ExpertCandidateStoreError(
                "composition source package differs from its candidate commit"
            )
        source_closure = ExpertCandidateStore._parse_closure(source_payloads)
        if commit.candidate_id != source_closure.manifest.candidate_id:
            raise ExpertCandidateStoreError(
                "composition source commit names another candidate"
            )
        reduction_source = ExpertCompositionReductionSource(
            source_reference=source_reference,
            validation_context=source_closure.validation_context,
            patch=source_closure.patch,
            candidate_tree=source_closure.candidate_tree,
            repository_map=source_closure.repository_map,
            module_contracts=source_closure.module_contracts,
            candidate_contents=source_closure.candidate_contents,
        )
        return ExpertCompositionSourceProvenance(
            candidate_manifest=source_closure.manifest,
            candidate_commit_record=commit,
            validation_context=source_closure.validation_context,
            reduction_source=reduction_source,
            parent_files=source_closure.parent_files,
            agent_derivation=source_closure.derivation,
            sanitation_report=source_closure.sanitation_report,
        )

    @staticmethod
    def _composition_source_root(candidate_id: str) -> str:
        require_content_id(candidate_id, "composition provenance candidate")
        return f"{_COMPOSITION_SOURCE_PROVENANCE_ROOT}/{candidate_id.rsplit(':', 1)[1]}"

    @staticmethod
    def _parse_contract_tuple(
        payload: bytes,
        contract_type: type[StrictContract],
        name: str,
    ) -> tuple:
        parsed = parse_json_bytes(payload)
        if not isinstance(parsed, list):
            raise ExpertCandidateStoreError(f"candidate {name} must be an array")
        return tuple(contract_type.from_dict(item) for item in parsed)

    def _candidate_path(self, candidate_id: str) -> Path:
        require_content_id(candidate_id, "candidate_id")
        return self.object_root / candidate_id.rsplit(":", 1)[1]

    def _prepare_layout(self) -> None:
        if os.path.lexists(self.root):
            self._validate_private_directory(self.root, "candidate store root")
        else:
            os.mkdir(self.root, mode=0o700)
            self._fsync_directory(self.state_root)
        for path in (self.object_root, self.staging_root):
            if os.path.lexists(path):
                self._validate_private_directory(path, "candidate store directory")
            else:
                os.mkdir(path, mode=0o700)
        lock_path = self.root / "candidate.lock"
        if not os.path.lexists(lock_path):
            descriptor = os.open(
                lock_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
            )
            os.close(descriptor)
        self._validate_private_regular(lock_path, "candidate store lock")
        self._fsync_directory(self.root)

    def _recover_staging(self) -> None:
        entries = tuple(sorted(self.staging_root.iterdir()))
        for entry in entries:
            if _STAGING_PATTERN.fullmatch(entry.name) is None or entry.is_symlink():
                raise ExpertCandidateStoreError(
                    "candidate staging contains an invalid entry"
                )
            identity = restricted_directory_identity(
                self.staging_root,
                entry.name,
                ExpertCandidateStoreError,
            )
            remove_restricted_directory(
                self.staging_root,
                entry.name,
                identity,
                ExpertCandidateStoreError,
            )

    def _exclusive_lock(self):
        return _CandidateStoreLock(self.root / "candidate.lock", exclusive=True)

    def _shared_lock(self):
        return _CandidateStoreLock(self.root / "candidate.lock", exclusive=False)

    @staticmethod
    def _write_private_file(root: Path, relative_path: str, payload: bytes) -> None:
        path = PurePosixPath(relative_path)
        if (
            path.is_absolute()
            or path == PurePosixPath(".")
            or ".." in path.parts
            or path.as_posix() != relative_path
        ):
            raise ExpertCandidateStoreError("candidate package path is invalid")
        output = root.joinpath(*path.parts)
        current = root
        for part in path.parts[:-1]:
            current /= part
            if not os.path.lexists(current):
                os.mkdir(current, mode=0o700)
            ExpertCandidateStore._validate_private_directory(
                current,
                "candidate package directory",
            )
        descriptor = os.open(
            output,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _read_private_tree(root: Path) -> dict[str, bytes]:
        ExpertCandidateStore._validate_private_directory(
            root,
            "candidate package root",
        )
        with ExitStack() as descriptors:
            root_descriptor = os.open(
                root,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            )
            descriptors.callback(os.close, root_descriptor)
            opened = os.fstat(root_descriptor)
            opened_identity = (opened.st_dev, opened.st_ino)
            files = ExpertCandidateStore._read_private_directory(
                root_descriptor,
                PurePosixPath("."),
            )
            current = os.stat(root, follow_symlinks=False)
            if (current.st_dev, current.st_ino) != opened_identity:
                raise ExpertCandidateStoreError(
                    "candidate package root changed during read"
                )
        return files

    @staticmethod
    def _read_private_directory(
        directory_descriptor: int,
        relative_root: PurePosixPath,
    ) -> dict[str, bytes]:
        with os.scandir(directory_descriptor) as iterator:
            entries = tuple(
                sorted(
                    (
                        (entry.name, entry.stat(follow_symlinks=False))
                        for entry in iterator
                    ),
                    key=lambda item: item[0],
                )
            )
        files: dict[str, bytes] = {}
        for name, expected in entries:
            current = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            if (current.st_dev, current.st_ino) != (
                expected.st_dev,
                expected.st_ino,
            ):
                raise ExpertCandidateStoreError(
                    "candidate package entry changed during read"
                )
            relative_path = (
                PurePosixPath(name)
                if relative_root == PurePosixPath(".")
                else relative_root / name
            )
            if stat.S_ISDIR(expected.st_mode):
                if expected.st_mode & (0o077 | stat.S_ISUID | stat.S_ISGID):
                    raise ExpertCandidateStoreError(
                        "candidate package directory is not private"
                    )
                with ExitStack() as child_descriptors:
                    child_descriptor = os.open(
                        name,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                        dir_fd=directory_descriptor,
                    )
                    child_descriptors.callback(os.close, child_descriptor)
                    opened = os.fstat(child_descriptor)
                    if (opened.st_dev, opened.st_ino) != (
                        expected.st_dev,
                        expected.st_ino,
                    ):
                        raise ExpertCandidateStoreError(
                            "candidate package directory changed during read"
                        )
                    files.update(
                        ExpertCandidateStore._read_private_directory(
                            child_descriptor,
                            relative_path,
                        )
                    )
                continue
            if (
                not stat.S_ISREG(expected.st_mode)
                or expected.st_nlink != 1
                or expected.st_mode & (0o077 | stat.S_ISUID | stat.S_ISGID)
            ):
                raise ExpertCandidateStoreError(
                    "candidate package entry is not a private independent file"
                )
            descriptor = os.open(
                name,
                os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=directory_descriptor,
            )
            with os.fdopen(descriptor, "rb") as handle:
                opened = os.fstat(handle.fileno())
                if (
                    (opened.st_dev, opened.st_ino) != (expected.st_dev, expected.st_ino)
                    or not stat.S_ISREG(opened.st_mode)
                    or opened.st_nlink != 1
                    or opened.st_mode & (0o077 | stat.S_ISUID | stat.S_ISGID)
                ):
                    raise ExpertCandidateStoreError(
                        "candidate package file changed during read"
                    )
                files[relative_path.as_posix()] = handle.read()
        return files

    @staticmethod
    def _validate_state_root(path: Path) -> None:
        if (
            not path.is_absolute()
            or path != Path(os.path.abspath(path))
            or path in {Path("/"), Path.home()}
            or path.is_symlink()
            or not path.is_dir()
            or path.resolve() != path
        ):
            raise ExpertCandidateStoreError(
                "candidate state root must be an authorized real directory"
            )
        if path.stat().st_mode & (0o077 | stat.S_ISUID | stat.S_ISGID):
            raise ExpertCandidateStoreError("candidate state root must be private")

    @staticmethod
    def _validate_private_directory(path: Path, name: str) -> None:
        if path.is_symlink() or not path.is_dir():
            raise ExpertCandidateStoreError(f"{name} must be a real directory")
        metadata = path.stat(follow_symlinks=False)
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_mode & (
            0o077 | stat.S_ISUID | stat.S_ISGID
        ):
            raise ExpertCandidateStoreError(f"{name} must be a private directory")

    @staticmethod
    def _validate_private_regular(path: Path, name: str) -> None:
        if path.is_symlink() or not path.is_file():
            raise ExpertCandidateStoreError(f"{name} must be a regular file")
        metadata = path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_mode & (0o077 | stat.S_ISUID | stat.S_ISGID)
        ):
            raise ExpertCandidateStoreError(
                f"{name} must be a private independent file"
            )

    @staticmethod
    def _fsync_tree(root: Path) -> None:
        directories = tuple(
            sorted(
                (path for path in root.rglob("*") if path.is_dir()),
                key=lambda path: len(path.parts),
                reverse=True,
            )
        )
        for directory in directories:
            ExpertCandidateStore._fsync_directory(directory)
        ExpertCandidateStore._fsync_directory(root)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        os.fsync(descriptor)
        os.close(descriptor)

    @staticmethod
    def _rename_directory_no_replace(source: Path, destination: Path) -> None:
        libc = ctypes.CDLL(None, use_errno=True)
        if not hasattr(libc, "renameat2"):
            raise ExpertCandidateStoreError(
                "atomic no-replace candidate publication is unavailable"
            )
        rename_at2 = libc.renameat2
        rename_at2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename_at2.restype = ctypes.c_int
        result = rename_at2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            _RENAME_NOREPLACE,
        )
        if result != 0:
            error_number = ctypes.get_errno()
            raise OSError(
                error_number,
                f"candidate publication failed: {errno.errorcode.get(error_number)}",
            )


class _SealedExpertCompositionAdmission:
    """One-shot process-local permission to persist one fenced composition."""

    __slots__ = (
        "_authority",
        "_approval_lease",
        "_commit_record",
        "_consumed",
        "_freshness_context",
        "_owner_process_id",
        "_package_files",
        "_snapshot",
        "_store",
    )

    def __init__(
        self,
        *,
        seal: object,
        store: ExpertCandidateStore,
        authority: ExpertCompositionAdmissionAuthority,
        approval_lease: ExpertCompositionApprovalLease,
        snapshot: ExpertCandidateClosure,
        package_files: dict[str, bytes],
        commit_record: ExpertCandidateCommitRecord,
        freshness_context: object,
    ) -> None:
        if seal is not _SEALED_COMPOSITION_ADMISSION:
            raise ExpertCandidateStoreError(
                "composition admission authority is not store sealed"
            )
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_authority", authority)
        object.__setattr__(self, "_approval_lease", approval_lease)
        object.__setattr__(self, "_owner_process_id", os.getpid())
        object.__setattr__(self, "_snapshot", snapshot)
        object.__setattr__(self, "_package_files", dict(package_files))
        object.__setattr__(self, "_commit_record", commit_record)
        object.__setattr__(self, "_freshness_context", freshness_context)
        object.__setattr__(self, "_consumed", False)

    def __setattr__(self, name, value) -> None:
        raise ExpertCandidateStoreError("composition admission authority is immutable")

    def __reduce__(self):
        raise ExpertCandidateStoreError(
            "composition admission authority cannot be serialized"
        )

    def __reduce_ex__(self, protocol):
        raise ExpertCandidateStoreError(
            "composition admission authority cannot be serialized"
        )

    def _consume(
        self,
        *,
        store: ExpertCandidateStore,
        authority: object,
    ) -> tuple[
        ExpertCandidateClosure,
        dict[str, bytes],
        ExpertCandidateCommitRecord,
        object,
        ExpertCompositionApprovalLease,
    ]:
        if (
            self._consumed
            or self._owner_process_id != os.getpid()
            or self._store is not store
            or self._authority is not authority
        ):
            raise ExpertCandidateStoreError(
                "composition admission authority is consumed or foreign"
            )
        object.__setattr__(self, "_consumed", True)
        return (
            self._snapshot,
            dict(self._package_files),
            self._commit_record,
            self._freshness_context,
            self._approval_lease,
        )


class _CandidateStoreLock:
    def __init__(self, path: Path, *, exclusive: bool, create: bool = False):
        self.path = path
        self.exclusive = exclusive
        self.create = create
        self.handle = None

    def __enter__(self):
        flags = os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC
        if self.create:
            flags |= os.O_CREAT
        descriptor = os.open(
            self.path,
            flags,
            0o600,
        )
        self.handle = os.fdopen(descriptor, "r+b")
        metadata = os.fstat(self.handle.fileno())
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_mode & (0o077 | stat.S_ISUID | stat.S_ISGID)
        ):
            self.handle.close()
            raise ExpertCandidateStoreError(
                "candidate lock must be a private independent file"
            )
        fcntl.flock(
            self.handle.fileno(),
            fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH,
        )
        return self

    def __exit__(self, exception_type, exception, traceback):
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()
        self.handle = None
        return False
