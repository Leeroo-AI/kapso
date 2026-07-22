from __future__ import annotations

from dataclasses import replace

import pytest

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    content_id,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    EvaluationFingerprint,
    ObjectiveDirection,
    TaskAdapterContextBinding,
    TaskAdapterManifest,
    TaskAdapterPackagePin,
    TaskAdapterRuntimeContract,
    TaskContextBinding,
    TaskEvaluatorBinding,
    TaskEvaluatorMetricComparisonBinding,
)
from kapso.cross_run.expert.promotion_contracts import (
    ExpertReleaseMatrixAdapterAuthority,
    ExpertReleaseMatrixComparisonRow,
    ExpertReleaseMatrixContractError,
    ExpertReleaseMatrixEvaluationCell,
    ExpertReleaseMatrixEvaluationPlan,
    ExpertReleaseMatrixMode,
    ExpertReleaseMatrixProvenanceBinding,
    ExpertReleaseMatrixProvenanceKind,
    ExpertReleaseMatrixReport,
)
from kapso.cross_run.task_adapters import (
    TaskAdapterVerificationReceipt,
    task_adapter_binding_id,
)


def _id(namespace: str, label: str) -> str:
    return content_id(namespace, {"label": label})


def _digest(label: str) -> str:
    return tree_or_blob_digest(label.encode("utf-8"))


def _remint(record, **changes):
    values = record.to_dict()
    values.pop(record.IDENTITY_FIELD)
    values.update(changes)
    return type(record).mint(**values)


def _comparison_binding(metric_name: str) -> TaskEvaluatorMetricComparisonBinding:
    return TaskEvaluatorMetricComparisonBinding(
        evaluator_fingerprint=_digest("release-evaluator"),
        metric_name=metric_name,
        objective_direction=ObjectiveDirection.MAXIMIZE,
        comparison_dimension_id=metric_name.removesuffix("_score"),
        comparison_scale=0.5,
    )


def _manifest(
    metric_names: tuple[str, ...] = ("quality_score", "robustness_score"),
) -> TaskAdapterManifest:
    bindings = tuple(
        sorted(
            (_comparison_binding(metric_name) for metric_name in metric_names),
            key=lambda binding: (
                binding.evaluator_fingerprint,
                binding.metric_name,
            ),
        )
    )
    return TaskAdapterManifest.mint(
        task_adapter_id="test_adapter",
        scope_contract_id=_id("expert-scope-contract", "scope"),
        task_family_id="test_family",
        publisher_attestation={"issuer": "test", "signature": "signed"},
        task_evaluator=TaskEvaluatorBinding(
            protocol_version="kapso.task_evaluator.v1",
            executable_path="adapter.py",
            supported_evaluator_fingerprints=(_digest("release-evaluator"),),
            metric_comparison_bindings=bindings,
        ),
        context_binding=TaskAdapterContextBinding(consumed_dimension_ids=()),
        source_tree_ref="task-adapter.tar.zst",
        tree_hash=_digest("adapter-tree"),
        runtime=TaskAdapterRuntimeContract(
            runtime_protocol_version="kapso.task_adapter_runtime.v1",
            image_repository="registry.example/kapso/task-adapter-runtime",
            image_manifest_digest=_digest("runtime-image"),
            image_config_digest=_digest("runtime-config"),
            dependency_lock_path="requirements.lock",
            dependency_lock_digest=_digest("runtime-lock"),
            operating_system="linux",
            architecture="amd64",
            architecture_variant=None,
            environment={"LANG": "C.UTF-8", "PATH": "/usr/bin:/bin"},
        ),
        sanitation_report_id=_id("task-adapter-sanitation", "adapter"),
        validation_refs=("validation.adapter_smoke",),
    )


def _verification_receipt(
    manifest: TaskAdapterManifest,
    **overrides,
) -> TaskAdapterVerificationReceipt:
    proof_refs = {manifest.sanitation_report_id, *manifest.validation_refs}
    values = {
        "task_adapter_manifest_id": manifest.task_adapter_manifest_id,
        "full_manifest_digest": tree_or_blob_digest(manifest.to_json_bytes()),
        "publisher_attestation_digest": tree_or_blob_digest(
            canonical_json_bytes(manifest.publisher_attestation)
        ),
        "source_extraction_receipt_id": _id(
            "source-archive-extraction-receipt", "adapter"
        ),
        "source_archive_ref": manifest.source_tree_ref,
        "source_archive_digest": _digest("source-archive"),
        "source_tree_hash": manifest.tree_hash,
        "proof_object_digests": {
            proof_ref: _digest(f"proof-{proof_ref}") for proof_ref in proof_refs
        },
        "publisher_verification_digest": _digest("publisher-verification"),
        "verifier_id": "test_verifier",
        "verifier_version": "test.verifier.v1",
    }
    values.update(overrides)
    return TaskAdapterVerificationReceipt.mint(**values)


def _provider_dependency_ids(
    manifest: TaskAdapterManifest,
    receipt: TaskAdapterVerificationReceipt,
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                receipt.verification_receipt_id,
                receipt.source_extraction_receipt_id,
                manifest.sanitation_report_id,
                *receipt.proof_object_ids,
            }
        )
    )


def _authority(
    metric_names: tuple[str, ...] = ("quality_score", "robustness_score"),
) -> ExpertReleaseMatrixAdapterAuthority:
    manifest = _manifest(metric_names)
    receipt = _verification_receipt(manifest)
    return ExpertReleaseMatrixAdapterAuthority.mint(
        task_adapter_pin=TaskAdapterPackagePin(
            adapter_binding_id=task_adapter_binding_id(
                manifest.task_family_id,
                manifest.task_adapter_id,
            ),
            task_adapter_manifest_id=manifest.task_adapter_manifest_id,
            verification_receipt_id=receipt.verification_receipt_id,
        ),
        task_adapter_manifest=manifest,
        verification_receipt=receipt,
        task_adapter_dependency_ids=_provider_dependency_ids(manifest, receipt),
    )


def _context(
    authority: ExpertReleaseMatrixAdapterAuthority,
    label: str = "task",
) -> TaskContextBinding:
    manifest = authority.task_adapter_manifest
    return TaskContextBinding.mint(
        scope_contract_id=manifest.scope_contract_id,
        scope_id="test_scope",
        task_family_id=manifest.task_family_id,
        task_adapter_id=manifest.task_adapter_id,
        capability_tags=("modeling",),
        input_contract_fingerprint=_digest(f"input-{label}"),
        target_contract_fingerprint=_digest(f"target-{label}"),
        starting_artifact_refs=(f"artifact/{label}",),
        method_fingerprint=_digest(f"method-{label}"),
        toolchain_fingerprint=_digest("toolchain"),
        dependency_runtime_fingerprint=_digest("dependency-runtime"),
        budget_hardware_envelope={"accelerator": "test", "hours": 1},
        transfer_dimensions={"dataset_family": label},
    )


def _provenance(
    context: TaskContextBinding,
    label: str = "task",
    *,
    provenance_kind: ExpertReleaseMatrixProvenanceKind = (
        ExpertReleaseMatrixProvenanceKind.SOURCE_REPLAY
    ),
) -> ExpertReleaseMatrixProvenanceBinding:
    if provenance_kind is ExpertReleaseMatrixProvenanceKind.ADAPTER_CASE:
        case_id = _id("task-adapter-release-matrix-case", label)
        return ExpertReleaseMatrixProvenanceBinding.mint(
            provenance_kind=provenance_kind,
            task_context_binding=context,
            adapter_case_id=case_id,
            source_replay_selection_id=None,
            source_bundle_id=None,
            bundle_lineage_ids=(),
            source_episode_ids=(),
            context_materialization_receipt_id=None,
            starting_artifact_ids=(),
            exact_dependency_ids=tuple(
                sorted({context.task_context_binding_id, case_id})
            ),
        )
    source_bundle_id = _id("run-bundle", label)
    bundle_lineage_ids = (
        _id("run-bundle", f"{label}-ancestor"),
        source_bundle_id,
    )
    source_episode_ids = (_id("transfer-episode", label),)
    materialization_id = _id("expert-source-replay-context-materialization", label)
    starting_artifact_ids = (_id("source-replay-starting-artifact", label),)
    selection_id = _id("expert-source-replay-selection", label)
    return ExpertReleaseMatrixProvenanceBinding.mint(
        provenance_kind=provenance_kind,
        task_context_binding=context,
        adapter_case_id=None,
        source_replay_selection_id=selection_id,
        source_bundle_id=source_bundle_id,
        bundle_lineage_ids=bundle_lineage_ids,
        source_episode_ids=source_episode_ids,
        context_materialization_receipt_id=materialization_id,
        starting_artifact_ids=starting_artifact_ids,
        exact_dependency_ids=tuple(
            sorted(
                {
                    context.task_context_binding_id,
                    selection_id,
                    *bundle_lineage_ids,
                    *source_episode_ids,
                    materialization_id,
                    *starting_artifact_ids,
                }
            )
        ),
    )


def _fingerprint(
    metric_name: str,
    seeds: tuple[str, ...] = ("repeat_1", "repeat_2"),
) -> EvaluationFingerprint:
    return EvaluationFingerprint.mint(
        benchmark_id="release_benchmark",
        dataset_version="release_data_v1",
        split_version="release_split_v1",
        evaluator_fingerprint=_digest("release-evaluator"),
        metric_name=metric_name,
        objective_direction=ObjectiveDirection.MAXIMIZE,
        fidelity="full",
        fraction=1.0,
        seed_or_replicate_ids=seeds,
        aggregation_protocol="paired_replicates",
        judge_version=None,
    )


def _cell(
    authority: ExpertReleaseMatrixAdapterAuthority,
    provenance: ExpertReleaseMatrixProvenanceBinding,
    metric_name: str,
    *,
    mode: ExpertReleaseMatrixMode = ExpertReleaseMatrixMode.PARENT_COMPARISON,
    seeds: tuple[str, ...] = ("repeat_1", "repeat_2"),
) -> ExpertReleaseMatrixEvaluationCell:
    fingerprint = _fingerprint(metric_name, seeds)
    binding = next(
        binding
        for binding in authority.task_adapter_manifest.task_evaluator.metric_comparison_bindings
        if binding.metric_name == metric_name
    )
    parent_release_id = (
        None
        if mode is ExpertReleaseMatrixMode.BOOTSTRAP
        else _id("expert-base-release", "parent")
    )
    parent_tree_hash = (
        None if mode is ExpertReleaseMatrixMode.BOOTSTRAP else _digest("parent-tree")
    )
    dependencies = {
        _id("expert-validation-attempt", "attempt"),
        _id("expert-candidate", "candidate"),
        authority.adapter_authority_id,
        provenance.provenance_binding_id,
        provenance.task_context_binding.task_context_binding_id,
        provenance.independence_identity_id,
        fingerprint.evaluation_fingerprint_id,
    }
    if parent_release_id is not None:
        dependencies.add(parent_release_id)
    return ExpertReleaseMatrixEvaluationCell.mint(
        mode=mode,
        validation_attempt_id=_id("expert-validation-attempt", "attempt"),
        candidate_id=_id("expert-candidate", "candidate"),
        candidate_tree_hash=_digest("candidate-tree"),
        parent_release_id=parent_release_id,
        parent_tree_hash=parent_tree_hash,
        adapter_authority_id=authority.adapter_authority_id,
        provenance_binding_id=provenance.provenance_binding_id,
        task_context_binding=provenance.task_context_binding,
        independence_identity_id=provenance.independence_identity_id,
        evaluation_fingerprint=fingerprint,
        metric_comparison_binding=binding,
        exact_dependency_ids=tuple(sorted(dependencies)),
    )


def _plan(
    authorities: tuple[ExpertReleaseMatrixAdapterAuthority, ...],
    provenances: tuple[ExpertReleaseMatrixProvenanceBinding, ...],
    cells: tuple[ExpertReleaseMatrixEvaluationCell, ...],
    *,
    mode: ExpertReleaseMatrixMode = ExpertReleaseMatrixMode.PARENT_COMPARISON,
) -> ExpertReleaseMatrixEvaluationPlan:
    ordered_authorities = tuple(
        sorted(authorities, key=lambda authority: authority.canonical_key)
    )
    ordered_provenances = tuple(
        sorted(provenances, key=lambda provenance: provenance.canonical_key)
    )
    ordered_cells = tuple(sorted(cells, key=lambda cell: cell.canonical_key))
    internal_ids = {
        *(authority.adapter_authority_id for authority in ordered_authorities),
        *(provenance.provenance_binding_id for provenance in ordered_provenances),
        *(cell.evaluation_cell_id for cell in ordered_cells),
    }
    external_dependencies = {
        dependency_id
        for authority in ordered_authorities
        for dependency_id in authority.exact_dependency_ids
    }
    external_dependencies.update(
        dependency_id
        for provenance in ordered_provenances
        for dependency_id in provenance.exact_dependency_ids
    )
    external_dependencies.update(
        dependency_id
        for cell in ordered_cells
        for dependency_id in cell.exact_dependency_ids
        if dependency_id not in internal_ids
    )
    external_dependencies.update(
        {
            _id("expert-candidate-commit", "candidate"),
            _id("expert-scope-contract", "scope"),
            _id("expert-validation-policy", "policy"),
        }
    )
    return ExpertReleaseMatrixEvaluationPlan.mint(
        mode=mode,
        validation_attempt_id=ordered_cells[0].validation_attempt_id,
        candidate_id=ordered_cells[0].candidate_id,
        candidate_commit_record_id=_id("expert-candidate-commit", "candidate"),
        candidate_tree_hash=ordered_cells[0].candidate_tree_hash,
        scope_contract_id=_id("expert-scope-contract", "scope"),
        parent_release_id=ordered_cells[0].parent_release_id,
        parent_tree_hash=ordered_cells[0].parent_tree_hash,
        validation_policy_id=_id("expert-validation-policy", "policy"),
        configuration_fingerprint=_digest("configuration"),
        adapter_authorities=ordered_authorities,
        provenance_bindings=ordered_provenances,
        evaluation_cells=ordered_cells,
        external_dependency_ids=tuple(sorted(external_dependencies)),
    )


def _row(
    cell: ExpertReleaseMatrixEvaluationCell,
    provenance: ExpertReleaseMatrixProvenanceBinding,
    *,
    candidate_values: dict[str, float] | None = None,
    parent_values: dict[str, float] | None = None,
) -> ExpertReleaseMatrixComparisonRow:
    selected_candidate_values = candidate_values or {
        replicate_id: 0.6 + position / 100.0
        for position, replicate_id in enumerate(
            cell.evaluation_fingerprint.seed_or_replicate_ids
        )
    }
    selected_parent_values = parent_values
    if (
        selected_parent_values is None
        and cell.mode is ExpertReleaseMatrixMode.PARENT_COMPARISON
    ):
        selected_parent_values = {
            replicate_id: 0.4 + position / 100.0
            for position, replicate_id in enumerate(
                cell.evaluation_fingerprint.seed_or_replicate_ids
            )
        }
    return ExpertReleaseMatrixComparisonRow.mint(
        evaluation_cell_id=cell.evaluation_cell_id,
        candidate_observation_event_id=_id(
            (
                "task-evaluation-journal-event"
                if provenance.provenance_kind
                is ExpertReleaseMatrixProvenanceKind.ADAPTER_CASE
                else "source-replay-execution-journal-event"
            ),
            f"candidate-{cell.evaluation_cell_id}",
        ),
        parent_observation_event_id=(
            None
            if cell.mode is ExpertReleaseMatrixMode.BOOTSTRAP
            else _id(
                (
                    "task-evaluation-journal-event"
                    if provenance.provenance_kind
                    is ExpertReleaseMatrixProvenanceKind.ADAPTER_CASE
                    else "source-replay-execution-journal-event"
                ),
                f"parent-{cell.evaluation_cell_id}",
            )
        ),
        candidate_replicate_values=selected_candidate_values,
        parent_replicate_values=selected_parent_values,
    )


def _report(
    plan: ExpertReleaseMatrixEvaluationPlan,
    rows: tuple[ExpertReleaseMatrixComparisonRow, ...],
) -> ExpertReleaseMatrixReport:
    first_cell = plan.evaluation_cells[0]
    dependencies = {
        first_cell.validation_attempt_id,
        first_cell.candidate_id,
        _id("expert-candidate-commit", "candidate"),
        _id("expert-scope-contract", "scope"),
        _id("expert-validation-policy", "policy"),
        plan.evaluation_plan_id,
        *plan.exact_dependency_ids,
        *(row.comparison_row_id for row in rows),
        *(dependency_id for row in rows for dependency_id in row.exact_dependency_ids),
    }
    if first_cell.parent_release_id is not None:
        dependencies.add(first_cell.parent_release_id)
    return ExpertReleaseMatrixReport.mint(
        mode=plan.mode,
        validation_attempt_id=first_cell.validation_attempt_id,
        candidate_id=first_cell.candidate_id,
        candidate_commit_record_id=_id("expert-candidate-commit", "candidate"),
        candidate_tree_hash=first_cell.candidate_tree_hash,
        scope_contract_id=_id("expert-scope-contract", "scope"),
        parent_release_id=first_cell.parent_release_id,
        parent_tree_hash=first_cell.parent_tree_hash,
        validation_policy_id=_id("expert-validation-policy", "policy"),
        configuration_fingerprint=_digest("configuration"),
        evaluation_plan=plan,
        evidence_rows=rows,
        exact_evidence_input_ids=tuple(
            sorted({plan.evaluation_plan_id, *plan.external_dependency_ids})
        ),
        exact_dependency_ids=tuple(sorted(dependencies)),
    )


def _matrix(
    mode: ExpertReleaseMatrixMode = ExpertReleaseMatrixMode.PARENT_COMPARISON,
) -> tuple[
    ExpertReleaseMatrixAdapterAuthority,
    ExpertReleaseMatrixProvenanceBinding,
    ExpertReleaseMatrixEvaluationPlan,
    ExpertReleaseMatrixReport,
]:
    authority = _authority()
    provenance = _provenance(
        _context(authority),
        provenance_kind=(
            ExpertReleaseMatrixProvenanceKind.ADAPTER_CASE
            if mode is ExpertReleaseMatrixMode.BOOTSTRAP
            else ExpertReleaseMatrixProvenanceKind.SOURCE_REPLAY
        ),
    )
    cells = tuple(
        _cell(authority, provenance, metric_name, mode=mode)
        for metric_name in ("quality_score", "robustness_score")
    )
    plan = _plan((authority,), (provenance,), cells, mode=mode)
    rows = tuple(_row(cell, provenance) for cell in plan.evaluation_cells)
    return authority, provenance, plan, _report(plan, rows)


def test_parent_report_round_trips_self_contained_precommitted_plan():
    authority, provenance, plan, report = _matrix()

    assert report.parent_tree_hash == _digest("parent-tree")
    assert report.exact_evidence_input_ids == tuple(
        sorted({plan.evaluation_plan_id, *plan.external_dependency_ids})
    )
    assert authority.adapter_authority_id in report.exact_dependency_ids
    assert provenance.provenance_binding_id in report.exact_dependency_ids
    assert provenance.independence_identity_id == provenance.bundle_lineage_ids[0]
    assert all(
        cell.independence_identity_id == provenance.bundle_lineage_ids[0]
        for cell in plan.evaluation_cells
    )
    assert all(
        cell.evaluation_cell_id in report.exact_dependency_ids
        for cell in plan.evaluation_cells
    )
    assert b"raw_delta" not in report.to_json_bytes()
    assert ExpertReleaseMatrixReport.from_json_bytes(report.to_json_bytes()) == report


def test_bootstrap_plan_and_rows_forbid_parent_authority():
    _, provenance, plan, report = _matrix(ExpertReleaseMatrixMode.BOOTSTRAP)

    assert report.parent_release_id is None
    assert report.parent_tree_hash is None
    assert provenance.provenance_kind is ExpertReleaseMatrixProvenanceKind.ADAPTER_CASE
    assert provenance.bundle_lineage_ids == ()
    assert all(
        row.candidate_observation_event_id.startswith("task-evaluation-journal-event:")
        for row in report.evidence_rows
    )
    assert all(row.parent_replicate_values is None for row in report.evidence_rows)
    with pytest.raises(ExpertReleaseMatrixContractError, match="cannot name a parent"):
        replace(plan.evaluation_cells[0], parent_tree_hash=_digest("substituted"))


def test_parent_plan_mixes_source_replay_and_adapter_owned_cases():
    authority = _authority()
    source_provenance = _provenance(_context(authority, "source"), "source")
    adapter_provenance = _provenance(
        _context(authority, "anchor"),
        "anchor",
        provenance_kind=ExpertReleaseMatrixProvenanceKind.ADAPTER_CASE,
    )
    provenances = (source_provenance, adapter_provenance)
    cells = tuple(
        _cell(authority, provenance, metric_name)
        for provenance in provenances
        for metric_name in ("quality_score", "robustness_score")
    )
    plan = _plan((authority,), provenances, cells)
    provenance_by_id = {
        provenance.provenance_binding_id: provenance for provenance in provenances
    }
    rows = tuple(
        _row(cell, provenance_by_id[cell.provenance_binding_id])
        for cell in plan.evaluation_cells
    )

    assert _report(plan, rows).evidence_rows == rows
    source_row_position = next(
        position
        for position, cell in enumerate(plan.evaluation_cells)
        if provenance_by_id[cell.provenance_binding_id].provenance_kind
        is ExpertReleaseMatrixProvenanceKind.SOURCE_REPLAY
    )
    substituted_channel = _remint(
        rows[source_row_position],
        candidate_observation_event_id=_id(
            "task-evaluation-journal-event",
            "substituted-channel",
        ),
    )
    with pytest.raises(ExpertReleaseMatrixContractError, match="observation channel"):
        _report(
            plan,
            tuple(
                substituted_channel if position == source_row_position else row
                for position, row in enumerate(rows)
            ),
        )


def test_adapter_authority_binds_full_attestation_source_and_provider_projection():
    authority = _authority()
    manifest = authority.task_adapter_manifest
    rotated_attestation = replace(
        manifest,
        publisher_attestation={"issuer": "test", "signature": "rotated"},
    )
    with pytest.raises(ExpertReleaseMatrixContractError, match="exact package"):
        replace(authority, task_adapter_manifest=rotated_attestation)

    mismatched_receipt = _verification_receipt(
        manifest,
        source_tree_hash=_digest("different-source-tree"),
    )
    mismatched_pin = replace(
        authority.task_adapter_pin,
        verification_receipt_id=mismatched_receipt.verification_receipt_id,
    )
    with pytest.raises(ExpertReleaseMatrixContractError, match="exact package"):
        ExpertReleaseMatrixAdapterAuthority.mint(
            task_adapter_pin=mismatched_pin,
            task_adapter_manifest=manifest,
            verification_receipt=mismatched_receipt,
            task_adapter_dependency_ids=_provider_dependency_ids(
                manifest, mismatched_receipt
            ),
        )

    with pytest.raises(ExpertReleaseMatrixContractError, match="projection"):
        ExpertReleaseMatrixAdapterAuthority.mint(
            task_adapter_pin=authority.task_adapter_pin,
            task_adapter_manifest=manifest,
            verification_receipt=authority.verification_receipt,
            task_adapter_dependency_ids=tuple(
                dependency_id
                for dependency_id in authority.task_adapter_dependency_ids
                if dependency_id != manifest.sanitation_report_id
            ),
        )


def test_provenance_binding_rejects_unknown_sources_and_nonexact_closure():
    authority = _authority(("quality_score",))
    context = _context(authority)
    provenance = _provenance(context)

    with pytest.raises(ExpertReleaseMatrixContractError, match="wrong namespace"):
        replace(
            provenance,
            context_materialization_receipt_id=_id("untrusted-source", "source"),
        )

    without_starting_artifacts = _remint(
        provenance,
        starting_artifact_ids=(),
        exact_dependency_ids=tuple(
            dependency_id
            for dependency_id in provenance.exact_dependency_ids
            if dependency_id not in provenance.starting_artifact_ids
        ),
    )
    assert without_starting_artifacts.starting_artifact_ids == ()
    with pytest.raises(ExpertReleaseMatrixContractError, match="not exact"):
        replace(
            provenance,
            exact_dependency_ids=tuple(
                dependency_id
                for dependency_id in provenance.exact_dependency_ids
                if dependency_id != provenance.source_episode_ids[0]
            ),
        )


def test_plan_requires_exact_declared_metric_coverage_and_single_context_lineage():
    authority = _authority()
    context = _context(authority)
    provenance = _provenance(context)
    quality = _cell(authority, provenance, "quality_score")
    robustness = _cell(authority, provenance, "robustness_score")
    plan = _plan((authority,), (provenance,), (quality, robustness))

    with pytest.raises(ExpertReleaseMatrixContractError, match="metric coverage"):
        replace(plan, evaluation_cells=(quality,))

    alternate_provenance = _provenance(
        context,
        "alternate",
    )
    alternate_robustness = _cell(
        authority,
        alternate_provenance,
        "robustness_score",
    )
    with pytest.raises(ExpertReleaseMatrixContractError, match="multiple lineage"):
        _plan(
            (authority,),
            (provenance, alternate_provenance),
            (quality, alternate_robustness),
        )


def test_one_fingerprint_cell_owns_its_complete_replicate_map():
    _, _, plan, report = _matrix()

    assert len(plan.evaluation_cells) == 2
    assert all(
        tuple(row.candidate_replicate_values)
        == cell.evaluation_fingerprint.seed_or_replicate_ids
        for cell, row in zip(
            plan.evaluation_cells,
            report.evidence_rows,
            strict=True,
        )
    )
    first = report.evidence_rows[0]
    with pytest.raises(
        ExpertReleaseMatrixContractError, match="candidate replicate coverage"
    ):
        _report(
            plan,
            (
                _remint(
                    first,
                    candidate_replicate_values={"repeat_1": 0.6},
                ),
                report.evidence_rows[1],
            ),
        )
    with pytest.raises(ExpertReleaseMatrixContractError, match="paired replicate"):
        _report(
            plan,
            (
                _remint(first, parent_replicate_values=None),
                report.evidence_rows[1],
            ),
        )
    with pytest.raises(ExpertReleaseMatrixContractError, match="must be distinct"):
        _remint(
            first,
            parent_observation_event_id=first.candidate_observation_event_id,
        )


def test_report_rejects_missing_extra_reordered_or_substituted_plan_cells():
    _, _, plan, report = _matrix()

    with pytest.raises(ExpertReleaseMatrixContractError, match="exactly once"):
        _report(plan, report.evidence_rows[:1])
    with pytest.raises(ExpertReleaseMatrixContractError, match="exactly once"):
        _report(plan, tuple(reversed(report.evidence_rows)))
    substituted = _remint(
        report.evidence_rows[0],
        evaluation_cell_id=_id("expert-release-matrix-evaluation-cell", "fake"),
    )
    with pytest.raises(ExpertReleaseMatrixContractError, match="exactly once"):
        _report(
            plan,
            (substituted, report.evidence_rows[1]),
        )


def test_plan_and_report_dependency_closures_are_exact():
    _, _, plan, report = _matrix()
    extra = _id("unrelated-evidence", "extra")

    with pytest.raises(ExpertReleaseMatrixContractError, match="external dependency"):
        replace(
            plan,
            external_dependency_ids=tuple(
                sorted((*plan.external_dependency_ids, extra))
            ),
        )
    with pytest.raises(ExpertReleaseMatrixContractError, match="evidence input"):
        replace(
            report,
            exact_evidence_input_ids=tuple(
                dependency_id
                for dependency_id in report.exact_evidence_input_ids
                if dependency_id != plan.evaluation_plan_id
            ),
        )
    with pytest.raises(ExpertReleaseMatrixContractError, match="not exact"):
        replace(
            report,
            exact_dependency_ids=tuple(
                dependency_id
                for dependency_id in report.exact_dependency_ids
                if dependency_id != plan.evaluation_cells[0].evaluation_cell_id
            ),
        )


def test_replicate_values_fail_loud_on_nonfinite_and_signed_zero():
    _, _, plan, report = _matrix()
    first = report.evidence_rows[0]

    with pytest.raises(ValueError, match="finite"):
        replace(
            first,
            candidate_replicate_values={
                **first.candidate_replicate_values,
                "repeat_1": float("inf"),
            },
        )
    with pytest.raises(ExpertReleaseMatrixContractError, match="signed zero"):
        replace(
            first,
            candidate_replicate_values={
                **first.candidate_replicate_values,
                "repeat_1": -0.0,
            },
        )
    assert _report(plan, report.evidence_rows) == report
