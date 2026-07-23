from pathlib import Path

import pytest

import kapso.cross_run.knowledge.package as knowledge_package_module
from kapso.cross_run.canonical import (
    canonical_json_bytes,
    content_id,
    parse_json_bytes,
    tree_or_blob_digest,
)
from kapso.cross_run.catalog.store import (
    CatalogGenerationManifest,
    CatalogInputDelta,
)
from kapso.cross_run.catalog.reducer import CatalogFactSet
from kapso.cross_run.contracts import (
    AdmissionState,
    CatalogEntryState,
    ContractValidationError,
    MissingReferenceError,
    PriorIdea,
    RunBundle,
)
from kapso.cross_run.record_contracts import (
    BundleProjectionManifest,
    ExpertReleaseUseRevocation,
    ExpertReleaseUseRevocationKind,
)
from kapso.cross_run.record_registry import (
    CATALOG_FACT_RECORD_TYPES,
    KNOWLEDGE_RECORD_TYPES,
)
from kapso.cross_run.knowledge.package import (
    KnowledgeSnapshotPackage,
    KnowledgeSnapshotPackageBuilder,
    KnowledgeSnapshotPackageError,
)
from test_cross_run_admission import prior_idea, sanitation_report, scope_contract
from test_cross_run_contracts import build_records


def digest(value):
    return tree_or_blob_digest(value.encode("utf-8"))


def test_typed_record_registry_matches_catalog_reducer_and_knowledge_superset():
    assert set(CatalogFactSet._FIELD_BY_TYPE) == set(CATALOG_FACT_RECORD_TYPES.values())
    assert set(CATALOG_FACT_RECORD_TYPES.items()).issubset(
        KNOWLEDGE_RECORD_TYPES.items()
    )


def empty_generation(scope):
    return CatalogGenerationManifest.mint(
        scope_contract_id=scope.scope_contract_id,
        generation_number=0,
        parent_generation_id=None,
        configuration_fingerprint=digest("catalog-config"),
        fact_object_ids=(),
        derived_object_ids=(),
        applied_input_delta_ids=(),
        bundle_frontier={},
        active_entry_state_ids={},
    )


def populated_generation():
    scope = scope_contract()
    template = prior_idea("snapshot")
    bundle = next(record for record in build_records() if isinstance(record, RunBundle))
    report = sanitation_report(bundle.bundle_id, bundle.task_context_binding)
    idea = PriorIdea.mint(
        source_bundle_id=bundle.bundle_id,
        supersedes_projection_id=None,
        source={
            "scope_id": bundle.scope_id,
            "run_id": bundle.run_id,
            "campaign_id": bundle.campaign_id,
            "batch_id": template.source["batch_id"],
            "idea_id": template.source["idea_id"],
        },
        proposal=template.proposal,
        descriptor=template.descriptor,
        assumptions=template.assumptions,
        source_status=template.source_status,
        source_rationale=template.source_rationale,
        source_evidence_refs=template.source_evidence_refs,
        task_context_binding=bundle.task_context_binding,
        sanitation_report_id=report.report_id,
    )
    projection = BundleProjectionManifest.mint(
        source_bundle_id=bundle.bundle_id,
        sanitation_report_id=report.report_id,
        episode_ids=(),
        prior_idea_ids=(idea.prior_idea_id,),
        derivation_object_ids=(),
    )
    state = CatalogEntryState.mint(
        subject_payload_id=idea.prior_idea_id,
        catalog_generation=1,
        predecessor_state_id=None,
        configuration_fingerprint=digest("catalog-config"),
        admission_state=AdmissionState.ADMITTED,
        superseded_by_payload_ids=(),
        assertion_ids=(),
        revocation_ids=(),
        taint_source_ids=(),
    )
    fact_ids = tuple(
        sorted(
            (
                bundle.bundle_id,
                report.report_id,
                idea.prior_idea_id,
                projection.projection_manifest_id,
            )
        )
    )
    input_delta = CatalogInputDelta.mint(
        scope_contract_id=scope.scope_contract_id,
        operation_id="snapshot-populated-test",
        configuration_fingerprint=digest("catalog-config"),
        added_object_ids=fact_ids,
        dependency_closure_ids=fact_ids,
    )
    generation = CatalogGenerationManifest.mint(
        scope_contract_id=scope.scope_contract_id,
        generation_number=1,
        parent_generation_id=content_id("fixture", {"generation": 0}),
        configuration_fingerprint=digest("catalog-config"),
        fact_object_ids=fact_ids,
        derived_object_ids=(state.catalog_entry_state_id,),
        applied_input_delta_ids=(input_delta.input_delta_id,),
        bundle_frontier={
            f"{bundle.scope_id}/{bundle.run_id}/{bundle.campaign_id}": (
                bundle.bundle_id
            )
        },
        active_entry_state_ids={
            idea.prior_idea_id: state.catalog_entry_state_id,
        },
    )
    objects = {
        bundle.bundle_id: bundle.to_json_bytes(),
        report.report_id: report.to_json_bytes(),
        idea.prior_idea_id: idea.to_json_bytes(),
        projection.projection_manifest_id: projection.to_json_bytes(),
        state.catalog_entry_state_id: state.to_json_bytes(),
        input_delta.input_delta_id: input_delta.to_json_bytes(),
    }
    return scope, idea, state, generation, objects


def populated_generation_with_release_use_revocations():
    scope, idea, state, generation, objects = populated_generation()
    evidence_id = next(
        object_id
        for object_id in generation.fact_object_ids
        if object_id.startswith("run-bundle:")
    )
    revocations = tuple(
        ExpertReleaseUseRevocation.mint(
            scope_contract_id=scope.scope_contract_id,
            scope_id=scope.scope_id,
            release_id=content_id("expert-base-release", {"kind": kind.value}),
            release_publication_id=content_id(
                "github-publication",
                {"kind": kind.value},
            ),
            release_activation_witness_id=content_id(
                "github-artifact-activation-witness",
                {"kind": kind.value},
            ),
            kind=kind,
            reason_code=f"{kind.value}_regression",
            rationale=f"Observed a release-wide {kind.value} regression.",
            exact_evidence_refs=(evidence_id,),
            recorded_at="2026-07-23T00:00:00Z",
        )
        for kind in (
            ExpertReleaseUseRevocationKind.COMPATIBILITY,
            ExpertReleaseUseRevocationKind.PERFORMANCE,
        )
    )
    fact_ids = tuple(
        sorted(
            (
                *generation.fact_object_ids,
                *(revocation.revocation_id for revocation in revocations),
            )
        )
    )
    input_delta = CatalogInputDelta.mint(
        scope_contract_id=scope.scope_contract_id,
        operation_id="snapshot-release-use-revocations-test",
        configuration_fingerprint=generation.configuration_fingerprint,
        added_object_ids=fact_ids,
        dependency_closure_ids=fact_ids,
    )
    generation_with_revocations = CatalogGenerationManifest.mint(
        scope_contract_id=generation.scope_contract_id,
        generation_number=generation.generation_number,
        parent_generation_id=generation.parent_generation_id,
        configuration_fingerprint=generation.configuration_fingerprint,
        fact_object_ids=fact_ids,
        derived_object_ids=generation.derived_object_ids,
        applied_input_delta_ids=(input_delta.input_delta_id,),
        bundle_frontier=generation.bundle_frontier,
        active_entry_state_ids=generation.active_entry_state_ids,
    )
    objects_with_revocations = {
        object_id: payload
        for object_id, payload in objects.items()
        if object_id not in generation.applied_input_delta_ids
    }
    objects_with_revocations.update(
        {
            input_delta.input_delta_id: input_delta.to_json_bytes(),
            **{
                revocation.revocation_id: revocation.to_json_bytes()
                for revocation in revocations
            },
        }
    )
    return (
        scope,
        idea,
        state,
        generation_with_revocations,
        objects_with_revocations,
        revocations,
    )


def finalize(prepared, **overrides):
    fields = {
        "parent_snapshot_ids": (),
        "sanitation_policy_version": "kapso.sanitation.v1",
        "retrieval_policy_version": "kapso.retrieval.v1",
        "configuration_fingerprint": digest("knowledge-config"),
        "prompt_budget_policy": {"maximum_records": 24},
        "published_at": "2026-07-21T12:00:00Z",
        "publisher_attestation": {"issuer": "test-publisher"},
    }
    fields.update(overrides)
    return KnowledgeSnapshotPackageBuilder.finalize(prepared, **fields)


def test_explicit_empty_snapshot_uses_the_normal_verified_path(tmp_path):
    scope = scope_contract()
    prepared = KnowledgeSnapshotPackageBuilder.prepare_empty(
        scope,
        empty_generation(scope),
    )
    package = finalize(prepared)

    assert prepared.snapshot_kind == "EMPTY"
    assert package.manifest.admitted_episode_ids == ()
    assert package.manifest.proof_dependency_closure_ids == tuple(
        record["record_id"] for record in prepared.record_envelopes
    )

    destination = (tmp_path / "materialized").absolute()
    package.materialize(destination)
    reopened = KnowledgeSnapshotPackage.open(destination)
    assert reopened.files == package.files
    assert reopened.manifest.snapshot_id == package.manifest.snapshot_id


def test_snapshot_is_order_independent_and_contains_complete_catalog_records():
    scope, idea, state, generation, objects = populated_generation()
    forward = KnowledgeSnapshotPackageBuilder.prepare(
        scope,
        generation,
        objects.__getitem__,
    )
    reversed_objects = dict(reversed(tuple(objects.items())))
    reverse = KnowledgeSnapshotPackageBuilder.prepare(
        scope,
        generation,
        reversed_objects.__getitem__,
    )

    first = finalize(forward)
    second = finalize(reverse)

    assert first.files == second.files
    assert first.manifest.snapshot_id == second.manifest.snapshot_id
    assert first.retrieval_root_ids == (idea.prior_idea_id,)
    assert state.catalog_entry_state_id in first.manifest.proof_dependency_closure_ids
    assert (
        canonical_json_bytes(first.record_by_id(idea.prior_idea_id)["payload"])
        == idea.to_json_bytes()
    )


def test_release_use_revocations_are_a_complete_nonretrieval_policy_projection(
    tmp_path,
):
    (
        scope,
        idea,
        _,
        generation,
        objects,
        revocations,
    ) = populated_generation_with_release_use_revocations()
    expected_revocation_ids = tuple(
        sorted(revocation.revocation_id for revocation in revocations)
    )
    prepared = KnowledgeSnapshotPackageBuilder.prepare(
        scope,
        generation,
        objects.__getitem__,
    )
    reverse = KnowledgeSnapshotPackageBuilder.prepare(
        scope,
        generation,
        dict(reversed(tuple(objects.items()))).__getitem__,
    )

    assert prepared.active_expert_release_use_revocation_ids == (
        expected_revocation_ids
    )
    assert reverse.active_expert_release_use_revocation_ids == (expected_revocation_ids)
    assert prepared.included_revocation_ids == ()
    assert prepared.retrieval_root_ids == (idea.prior_idea_id,)
    assert set(expected_revocation_ids).isdisjoint(prepared.retrieval_root_ids)
    for revocation in revocations:
        assert prepared.proof_dependencies[revocation.revocation_id] == tuple(
            sorted(
                (
                    scope.scope_contract_id,
                    *revocation.exact_evidence_refs,
                )
            )
        )

    package = finalize(prepared)
    reverse_package = finalize(reverse)

    assert package.files == reverse_package.files
    assert package.manifest.active_expert_release_use_revocation_ids == (
        expected_revocation_ids
    )
    assert set(expected_revocation_ids).issubset(
        package.manifest.proof_dependency_closure_ids
    )

    destination = (tmp_path / "release-use-policy").absolute()
    package.materialize(destination)
    reopened = KnowledgeSnapshotPackage.open(destination)

    assert reopened.manifest.active_expert_release_use_revocation_ids == (
        expected_revocation_ids
    )
    assert reopened.prepared.active_expert_release_use_revocation_ids == (
        expected_revocation_ids
    )

    forged_payload = {
        key: value
        for key, value in package.manifest.to_dict().items()
        if key not in {"snapshot_id", "active_expert_release_use_revocation_ids"}
    }
    forged_manifest = type(package.manifest).mint(
        **forged_payload,
        active_expert_release_use_revocation_ids=(expected_revocation_ids[0],),
    )
    manifest_path = destination / "snapshot.json"
    manifest_path.chmod(0o644)
    manifest_path.write_bytes(forged_manifest.to_json_bytes())

    with pytest.raises(
        KnowledgeSnapshotPackageError,
        match="manifest differs from its exact catalog closure",
    ):
        KnowledgeSnapshotPackage.open(destination)


def test_snapshot_rejects_self_consistent_record_with_unknown_schema_field():
    scope, _, _, generation, objects = populated_generation()
    original_delta_id = generation.applied_input_delta_ids[0]
    forged_delta = dict(parse_json_bytes(objects[original_delta_id]))
    forged_delta["unexpected_field"] = "not part of CatalogInputDelta"
    forged_delta["input_delta_id"] = content_id(
        "catalog-input-delta",
        {key: value for key, value in forged_delta.items() if key != "input_delta_id"},
    )
    forged_delta_id = forged_delta["input_delta_id"]
    forged_generation = CatalogGenerationManifest.mint(
        scope_contract_id=generation.scope_contract_id,
        generation_number=generation.generation_number,
        parent_generation_id=generation.parent_generation_id,
        configuration_fingerprint=generation.configuration_fingerprint,
        fact_object_ids=generation.fact_object_ids,
        derived_object_ids=generation.derived_object_ids,
        applied_input_delta_ids=(forged_delta_id,),
        bundle_frontier=generation.bundle_frontier,
        active_entry_state_ids=generation.active_entry_state_ids,
    )
    forged_objects = {
        key: value for key, value in objects.items() if key != original_delta_id
    }
    forged_objects[forged_delta_id] = canonical_json_bytes(forged_delta)

    with pytest.raises(ContractValidationError, match="fields mismatch"):
        KnowledgeSnapshotPackageBuilder.prepare(
            scope,
            forged_generation,
            forged_objects.__getitem__,
        )


def test_snapshot_rejects_missing_or_corrupt_proof_bytes(tmp_path):
    scope, _, state, generation, objects = populated_generation()
    prepared = KnowledgeSnapshotPackageBuilder.prepare(
        scope,
        generation,
        objects.__getitem__,
    )
    package = finalize(prepared)
    destination = (tmp_path / "corrupt").absolute()
    package.materialize(destination)
    state_path = next(
        path
        for path in destination.rglob("*.json")
        if state.catalog_entry_state_id.rsplit(":", 1)[1] in path.name
    )
    state_path.chmod(0o644)
    state_path.write_bytes(b"{}")

    with pytest.raises(KnowledgeSnapshotPackageError, match="checksum mismatch"):
        KnowledgeSnapshotPackage.open(destination)


def test_snapshot_closes_catalog_inputs_and_entry_state_predecessors():
    scope, idea, predecessor, first_generation, objects = populated_generation()
    current = CatalogEntryState.mint(
        subject_payload_id=idea.prior_idea_id,
        catalog_generation=2,
        predecessor_state_id=predecessor.catalog_entry_state_id,
        configuration_fingerprint=digest("catalog-config"),
        admission_state=AdmissionState.ADMITTED,
        superseded_by_payload_ids=(),
        assertion_ids=(),
        revocation_ids=(),
        taint_source_ids=(),
    )
    generation = CatalogGenerationManifest.mint(
        scope_contract_id=scope.scope_contract_id,
        generation_number=2,
        parent_generation_id=first_generation.catalog_generation_id,
        configuration_fingerprint=digest("catalog-config"),
        fact_object_ids=first_generation.fact_object_ids,
        derived_object_ids=(current.catalog_entry_state_id,),
        applied_input_delta_ids=first_generation.applied_input_delta_ids,
        bundle_frontier=first_generation.bundle_frontier,
        active_entry_state_ids={
            idea.prior_idea_id: current.catalog_entry_state_id,
        },
    )
    objects = {
        **objects,
        current.catalog_entry_state_id: current.to_json_bytes(),
    }

    package = finalize(
        KnowledgeSnapshotPackageBuilder.prepare(
            scope,
            generation,
            objects.__getitem__,
        )
    )

    assert package.manifest.entry_state_refs == (current.catalog_entry_state_id,)
    assert predecessor.catalog_entry_state_id in (
        package.manifest.proof_dependency_closure_ids
    )
    assert first_generation.applied_input_delta_ids[0] in (
        package.manifest.proof_dependency_closure_ids
    )


def test_snapshot_rejects_a_missing_typed_proof_dependency():
    scope, _, _, generation, objects = populated_generation()
    missing_report_id = next(
        object_id for object_id in objects if object_id.startswith("sanitation-report:")
    )
    remaining_fact_ids = tuple(
        object_id
        for object_id in generation.fact_object_ids
        if object_id != missing_report_id
    )
    input_delta = CatalogInputDelta.mint(
        scope_contract_id=scope.scope_contract_id,
        operation_id="snapshot-missing-dependency-test",
        configuration_fingerprint=generation.configuration_fingerprint,
        added_object_ids=remaining_fact_ids,
        dependency_closure_ids=remaining_fact_ids,
    )
    incomplete_generation = CatalogGenerationManifest.mint(
        scope_contract_id=scope.scope_contract_id,
        generation_number=generation.generation_number,
        parent_generation_id=generation.parent_generation_id,
        configuration_fingerprint=generation.configuration_fingerprint,
        fact_object_ids=remaining_fact_ids,
        derived_object_ids=generation.derived_object_ids,
        applied_input_delta_ids=(input_delta.input_delta_id,),
        bundle_frontier=generation.bundle_frontier,
        active_entry_state_ids=generation.active_entry_state_ids,
    )
    incomplete_objects = {
        object_id: payload
        for object_id, payload in objects.items()
        if object_id not in {missing_report_id, *generation.applied_input_delta_ids}
    }
    incomplete_objects[input_delta.input_delta_id] = input_delta.to_json_bytes()

    with pytest.raises(MissingReferenceError, match="typed proof dependency"):
        KnowledgeSnapshotPackageBuilder.prepare(
            scope,
            incomplete_generation,
            incomplete_objects.__getitem__,
        )


def test_materialization_refuses_an_existing_or_relative_target(tmp_path):
    scope = scope_contract()
    package = finalize(
        KnowledgeSnapshotPackageBuilder.prepare_empty(
            scope,
            empty_generation(scope),
        )
    )

    with pytest.raises(KnowledgeSnapshotPackageError, match="absolute normalized"):
        package.materialize(Path("relative"))
    with pytest.raises(KnowledgeSnapshotPackageError, match="already exists"):
        package.materialize(tmp_path.absolute())


def test_materialization_never_replaces_a_destination_created_during_commit(
    tmp_path,
    monkeypatch,
):
    scope = scope_contract()
    package = finalize(
        KnowledgeSnapshotPackageBuilder.prepare_empty(
            scope,
            empty_generation(scope),
        )
    )
    destination = (tmp_path / "concurrent-owner").absolute()
    publish_no_replace = knowledge_package_module._rename_directory_no_replace

    def create_destination_then_publish(source, target):
        target.mkdir()
        publish_no_replace(source, target)

    monkeypatch.setattr(
        knowledge_package_module,
        "_rename_directory_no_replace",
        create_destination_then_publish,
    )

    with pytest.raises(KnowledgeSnapshotPackageError, match="already exists"):
        package.materialize(destination)

    assert destination.is_dir()
    assert tuple(destination.iterdir()) == ()
