from __future__ import annotations

from dataclasses import replace

import pytest

import test_expert_release_matrix_reservation as reservation_fixture_module
from kapso.core.config import load_config
from kapso.cross_run.canonical import source_tree_digest, tree_or_blob_digest
from kapso.cross_run.contracts import ExpertPromotionState
from kapso.cross_run.expert.release import (
    EXPERT_RELEASE_EVIDENCE_MANIFEST_PATH,
    EXPERT_RELEASE_MANIFEST_PATH,
    ExpertReleaseAssembler,
    ExpertReleaseAssemblyError,
)
from kapso.cross_run.settings import CrossRunSettings
from kapso.cross_run.source_archives import (
    SourceArchiveError,
    build_deterministic_tar_zst,
)
from test_expert_promotion_decision import _settings
from test_expert_promotion_evidence import _bootstrap_prepared_with_store
from test_expert_promotion_stage import _completed_runtime
from test_expert_publication_eligibility import _coordinator
from kapso.cross_run.expert.promotion_stage import ExpertReleaseMatrixStageCoordinator

CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"


def _approved_bootstrap(tmp_path, monkeypatch):
    settings = _settings(minimum_replicates=1, minimum_pairs=1)
    monkeypatch.setattr(
        reservation_fixture_module,
        "_quality_only_validation_settings",
        lambda: settings,
    )
    validation_store, snapshot, prepared = _bootstrap_prepared_with_store(
        tmp_path,
        monkeypatch,
    )
    reservation, execution_store, completed = _completed_runtime(
        validation_store,
        snapshot,
        prepared,
    )
    matrix = ExpertReleaseMatrixStageCoordinator(
        validation_store=validation_store,
        execution_store=execution_store,
    ).publish_completed(
        completed_execution=completed,
        reservation_snapshot=reservation,
        prepared_request=prepared,
    )
    case = type(
        "ReleaseCase",
        (),
        {
            "validation_store": validation_store,
            "matrix_commit": matrix,
            "prepared": prepared,
        },
    )()
    approval = _coordinator(case).coordinator.publish(
        candidate_id=matrix.snapshot.state.candidate_id,
        release_matrix_stage_result_id=matrix.stage_result.stage_result_record_id,
    )
    return validation_store, matrix, approval


def test_release_assembly_is_exact_deterministic_and_approval_only(
    tmp_path,
    monkeypatch,
):
    validation_store, matrix, approval = _approved_bootstrap(tmp_path, monkeypatch)
    candidate_store = validation_store.reducer.candidate_store
    stored_candidate = candidate_store.read(approval.snapshot.state.candidate_id)
    settings = CrossRunSettings.from_dict(
        load_config(CANONICAL_CONFIG_PATH)["cross_run"]
    )
    assembler = ExpertReleaseAssembler(
        candidate_store=candidate_store,
        validation_store=validation_store,
        expert_settings=candidate_store.validator.settings,
        github_settings=settings.github,
    )

    first = assembler.build(
        candidate_id=stored_candidate.closure.manifest.candidate_id,
    )
    second = assembler.build(
        candidate_id=stored_candidate.closure.manifest.candidate_id,
    )

    assert approval.snapshot.state.promotion_state is ExpertPromotionState.APPROVED
    assert first.manifest == second.manifest
    assert first.source_archive == second.source_archive
    assert first.evidence_archive == second.evidence_archive
    assert first.control_archive == second.control_archive
    assert first.manifest.candidate_tree_hash == source_tree_digest(
        {
            path: (tree_or_blob_digest(payload), mode, len(payload))
            for path, (payload, mode) in first.source_files.items()
        }
    )
    assert first.evidence_manifest.evidence_manifest_id in (
        first.manifest.dependency_closure_ids
    )
    assert EXPERT_RELEASE_EVIDENCE_MANIFEST_PATH in first.evidence_files
    assert EXPERT_RELEASE_MANIFEST_PATH in first.publication_files
    assert not any(
        "artifacts/" in path
        or "workspace-delta" in path
        or "operation-receipt" in path
        or "expert-evaluator-result" in path
        for path in first.evidence_files
    )

    changed_source_files = dict(first.source_files)
    changed_path = next(iter(changed_source_files))
    payload, mode = changed_source_files[changed_path]
    changed_source_files[changed_path] = (payload + b"mutated", mode)
    with pytest.raises(ExpertReleaseAssemblyError):
        assembler.verify(replace(first, source_files=changed_source_files))
    unrelated_dependency = "unrelated-release-evidence:sha256:" + "f" * 64
    with pytest.raises(ValueError, match="not exact"):
        replace(
            first.manifest,
            dependency_closure_ids=tuple(
                sorted(
                    {
                        *first.manifest.dependency_closure_ids,
                        unrelated_dependency,
                    }
                )
            ),
        )


@pytest.mark.parametrize(
    "files",
    (
        {"unsafe\\path": (b"payload", "100644")},
        {"collision": (b"file", "100644"), "collision/child": (b"nested", "100644")},
        {".git/config": (b"metadata", "100644")},
        {".gitmodules": (b"metadata", "100644")},
    ),
)
def test_deterministic_archive_writer_rejects_reader_incompatible_paths(files):
    settings = CrossRunSettings.from_dict(
        load_config(CANONICAL_CONFIG_PATH)["cross_run"]
    )
    with pytest.raises(SourceArchiveError, match="path closure"):
        build_deterministic_tar_zst(
            files,
            compression_level=settings.expert.release_archive_compression_level,
            zstd_window_size_bytes=settings.github.zstd_window_size_bytes,
        )
