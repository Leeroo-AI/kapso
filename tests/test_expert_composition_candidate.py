from __future__ import annotations

from dataclasses import fields, replace

import pytest

from kapso.cross_run.canonical import (
    content_id,
    source_tree_digest,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    ExpertCandidateCommitRecord,
    ExpertCandidateManifest,
    ExpertCandidatePatch,
    ExpertCandidatePatchChange,
    ExpertCandidateDerivationKind,
    ExpertSourceTreeManifest,
    ExpertValidationTrack,
)
from kapso.cross_run.expert.book import expert_control_paths
from kapso.cross_run.expert.candidate_derivations import (
    ExpertCompositionSourceProvenance,
    ExpertDeterministicCompositionDerivation,
    ExpertDeterministicCompositionDerivationRecord,
)
from kapso.cross_run.expert.candidate_package import (
    AGENT_TRIGGER_DECISION_PACKAGE_PATH,
)
from kapso.cross_run.expert.composition import ExpertCompositionReductionSource
from kapso.cross_run.expert.composition_candidate import (
    ExpertCompositionCandidateError,
    project_deterministic_composition_candidate,
)
from kapso.cross_run.expert.composition_contracts import (
    ExpertCompositionPlan,
    expert_composition_configuration_fingerprint,
)
from kapso.cross_run.expert.store import (
    ExpertCandidateStore,
    ExpertCandidateStoreError,
)
from test_expert_composition import _plan, _released_base, reducer_case
from test_expert_publication_eligibility import terminal_cases


def _remint(record, **changes):
    payload = {
        field.name: getattr(record, field.name)
        for field in fields(record)
        if field.name != record.IDENTITY_FIELD
    }
    payload.update(changes)
    return type(record).mint(**payload)


def _project(case):
    plan = _plan(case.parent_base, (case.source,))
    reduction = case.reducer.reduce(
        plan=plan,
        current_base=case.parent_base,
        sources=(case.source.reduction_source,),
    )
    closure = project_deterministic_composition_candidate(
        reduction=reduction,
        current_base=case.parent_base,
        approved_sources=(case.source,),
        sanitizer=case.source._resolver.candidate_store.validator.sanitizer,
    )
    return plan, reduction, closure


def test_clean_composition_projects_one_valid_behavioral_candidate(reducer_case):
    plan, reduction, closure = _project(reducer_case)
    source_closure = reducer_case.source.stored_candidate.closure

    assert closure.derivation.record.source_validation_context_ids == {
        source_closure.manifest.candidate_id: (
            source_closure.validation_context.validation_context_id
        )
    }
    assert closure.origin_principal_ids == source_closure.origin_principal_ids
    assert closure.manifest.derivation_kind is (
        ExpertCandidateDerivationKind.DETERMINISTIC_COMPOSITION
    )
    assert closure.manifest.ancestor_candidate_ids == (
        source_closure.manifest.candidate_id,
    )
    assert closure.validation_track is ExpertValidationTrack.BEHAVIORAL_CAPABILITY
    assert closure.patch == reduction.materialization.patch
    assert closure.candidate_tree == reduction.materialization.source_tree
    assert (
        closure.derivation.materialization.composition_assessment.composition_plan
        == (plan)
    )
    reducer_case.source._resolver.candidate_store.validator.validate(closure)

    _, _, replayed = _project(reducer_case)
    assert replayed == closure


def test_composition_candidate_package_codec_is_exact(reducer_case, tmp_path):
    _, _, closure = _project(reducer_case)
    store = ExpertCandidateStore(
        tmp_path / "candidates",
        tmp_path,
        reducer_case.source._resolver.candidate_store.validator,
    )

    payloads = store._package_files(closure)
    reopened = store._parse_closure(payloads)

    assert reopened == closure
    store.validator.validate_persisted(reopened)
    with pytest.raises(
        ExpertCandidateStoreError,
        match="sealed admission authority",
    ):
        store.persist(closure)


def test_composition_derivation_rejects_incomplete_source_provenance(reducer_case):
    _, _, closure = _project(reducer_case)
    record = closure.derivation.record

    with pytest.raises(ValueError, match="provenance keys"):
        ExpertDeterministicCompositionDerivationRecord.mint(
            composition_materialization_id=record.composition_materialization_id,
            source_validation_context_ids=record.source_validation_context_ids,
            source_origin_principal_ids={},
            source_dependency_ids=record.source_dependency_ids,
        )


def test_composition_reduction_source_rejects_another_validation_context(
    reducer_case,
):
    source = reducer_case.source.reduction_source
    reference = source.source_reference
    foreign_context_id = content_id(
        "expert-candidate-validation-context",
        {"foreign": "context"},
    )
    stable_authorities = set(reference.stable_authority_ids)
    stable_authorities.remove(reference.validation_context_ref)
    stable_authorities.add(foreign_context_id)
    foreign_reference = _remint(
        reference,
        validation_context_ref=foreign_context_id,
        stable_authority_ids=tuple(sorted(stable_authorities)),
    )

    with pytest.raises(ValueError, match="stable reference"):
        ExpertCompositionReductionSource(
            source_reference=foreign_reference,
            validation_context=source.validation_context,
            patch=source.patch,
            candidate_tree=source.candidate_tree,
            repository_map=source.repository_map,
            module_contracts=source.module_contracts,
            candidate_contents=source.candidate_contents,
        )


def test_composition_validator_rejects_partial_source_commit(reducer_case):
    _, _, closure = _project(reducer_case)
    derivation = closure.derivation
    provenance = derivation.source_provenance[0]
    source_reference = provenance.reduction_source.source_reference
    incomplete_checksums = dict(provenance.candidate_commit_record.file_checksums)
    del incomplete_checksums[AGENT_TRIGGER_DECISION_PACKAGE_PATH]
    incomplete_commit = ExpertCandidateCommitRecord.mint(
        candidate_id=provenance.candidate_id,
        file_checksums=incomplete_checksums,
    )
    source_authorities = set(source_reference.stable_authority_ids)
    source_authorities.remove(source_reference.candidate_commit_record_id)
    source_authorities.add(incomplete_commit.commit_record_id)
    incomplete_source_reference = _remint(
        source_reference,
        candidate_commit_record_id=incomplete_commit.commit_record_id,
        stable_authority_ids=tuple(sorted(source_authorities)),
    )
    incomplete_reduction_source = ExpertCompositionReductionSource(
        source_reference=incomplete_source_reference,
        validation_context=provenance.validation_context,
        patch=provenance.reduction_source.patch,
        candidate_tree=provenance.reduction_source.candidate_tree,
        repository_map=provenance.reduction_source.repository_map,
        module_contracts=provenance.reduction_source.module_contracts,
        candidate_contents=provenance.reduction_source.candidate_contents,
    )
    incomplete_provenance = ExpertCompositionSourceProvenance(
        candidate_manifest=provenance.candidate_manifest,
        candidate_commit_record=incomplete_commit,
        validation_context=provenance.validation_context,
        reduction_source=incomplete_reduction_source,
        parent_files=provenance.parent_files,
        agent_derivation=provenance.agent_derivation,
        sanitation_report=provenance.sanitation_report,
    )
    materialization = derivation.materialization
    plan = materialization.composition_assessment.composition_plan
    plan_authorities = {
        plan.scope_contract.scope_contract_id,
        plan.current_base.base_reference_id,
        *plan.current_base.stable_authority_ids,
        incomplete_source_reference.source_reference_id,
        *incomplete_source_reference.stable_authority_ids,
    }
    superseded_scope_id = plan.scope_contract.supersedes_scope_contract_id
    if superseded_scope_id is not None:
        plan_authorities.add(superseded_scope_id)
    incomplete_plan = _remint(
        plan,
        sources=(incomplete_source_reference,),
        stable_authority_ids=tuple(sorted(plan_authorities)),
    )
    assessment = materialization.composition_assessment
    incomplete_assessment = _remint(
        assessment,
        composition_plan=incomplete_plan,
        applicable_source_reference_ids=(
            incomplete_source_reference.source_reference_id,
        ),
        stable_authority_ids=tuple(
            sorted(
                {
                    incomplete_plan.composition_plan_id,
                    *incomplete_plan.stable_authority_ids,
                }
            )
        ),
    )
    incomplete_materialization = _remint(
        materialization,
        composition_assessment=incomplete_assessment,
        stable_authority_ids=tuple(
            sorted(
                {
                    incomplete_assessment.assessment_id,
                    *incomplete_assessment.stable_authority_ids,
                    materialization.parent_tree.source_tree_manifest_id,
                    materialization.patch.patch_id,
                    materialization.source_tree.source_tree_manifest_id,
                    materialization.repository_map.repository_map_id,
                    *(
                        module.module_contract_id
                        for module in materialization.module_contracts
                    ),
                }
            )
        ),
    )
    dependencies = tuple(
        sorted(
            {
                incomplete_plan.composition_plan_id,
                *incomplete_plan.stable_authority_ids,
                *derivation.record.source_validation_context_ids.values(),
            }
        )
    )
    incomplete_record = _remint(
        derivation.record,
        composition_materialization_id=incomplete_materialization.materialization_id,
        source_dependency_ids=dependencies,
    )
    incomplete_derivation = ExpertDeterministicCompositionDerivation(
        record=incomplete_record,
        materialization=incomplete_materialization,
        source_provenance=(incomplete_provenance,),
        parent_contents=derivation.parent_contents,
    )
    incomplete_manifest = _remint(
        closure.manifest,
        derivation_ref=incomplete_record.derivation_id,
        source_dependency_ids=dependencies,
    )
    incomplete_closure = replace(
        closure,
        manifest=incomplete_manifest,
        derivation=incomplete_derivation,
    )

    with pytest.raises(ValueError, match="source package differs"):
        reducer_case.source._resolver.candidate_store.validator.validate(
            incomplete_closure
        )


def test_composition_validator_rejects_materialization_substitution(reducer_case):
    _, _, closure = _project(reducer_case)
    materialization = closure.derivation.materialization
    altered_policy = "kapso.expert_composition.v2"
    original_plan = materialization.composition_assessment.composition_plan
    altered_plan = _remint(
        original_plan,
        composition_policy_version=altered_policy,
        configuration_fingerprint=expert_composition_configuration_fingerprint(
            composition_policy_version=altered_policy,
            composition_source_limit=original_plan.composition_source_limit,
            candidate_entry_limit=original_plan.candidate_entry_limit,
            candidate_byte_limit=original_plan.candidate_byte_limit,
        ),
    )
    assert type(altered_plan) is ExpertCompositionPlan
    altered_assessment = _remint(
        materialization.composition_assessment,
        composition_plan=altered_plan,
        stable_authority_ids=tuple(
            sorted(
                {
                    altered_plan.composition_plan_id,
                    *altered_plan.stable_authority_ids,
                }
            )
        ),
    )
    altered_materialization = _remint(
        materialization,
        composition_assessment=altered_assessment,
        stable_authority_ids=tuple(
            sorted(
                {
                    altered_assessment.assessment_id,
                    *altered_assessment.stable_authority_ids,
                    materialization.parent_tree.source_tree_manifest_id,
                    materialization.patch.patch_id,
                    materialization.source_tree.source_tree_manifest_id,
                    materialization.repository_map.repository_map_id,
                    *(
                        module.module_contract_id
                        for module in materialization.module_contracts
                    ),
                }
            )
        ),
    )

    with pytest.raises(ValueError, match="record differs"):
        ExpertDeterministicCompositionDerivation(
            record=closure.derivation.record,
            materialization=altered_materialization,
            source_provenance=closure.derivation.source_provenance,
            parent_contents=closure.derivation.parent_contents,
        )


def test_composition_validator_rejects_code_not_produced_by_reducer(reducer_case):
    _, _, closure = _project(reducer_case)
    controls = set(expert_control_paths(closure.module_contracts))
    change = next(
        change
        for change in closure.patch.changes
        if change.after is not None and change.relative_path not in controls
    )
    forged_contents = dict(closure.candidate_contents)
    forged_contents[change.relative_path] += b"\n# forged composition output\n"
    forged_descriptors = tuple(
        (
            replace(
                descriptor,
                digest=tree_or_blob_digest(forged_contents[descriptor.relative_path]),
                size=len(forged_contents[descriptor.relative_path]),
            )
            if descriptor.relative_path == change.relative_path
            else descriptor
        )
        for descriptor in closure.candidate_tree.files
    )
    forged_tree = ExpertSourceTreeManifest.mint(
        tree_hash=source_tree_digest(
            {
                descriptor.relative_path: (
                    descriptor.digest,
                    descriptor.mode,
                    descriptor.size,
                )
                for descriptor in forged_descriptors
            }
        ),
        files=forged_descriptors,
    )
    parent_files = {
        descriptor.relative_path: descriptor for descriptor in closure.parent_files
    }
    candidate_files = {
        descriptor.relative_path: descriptor for descriptor in forged_descriptors
    }
    forged_patch = ExpertCandidatePatch.mint(
        parent_tree_hash=closure.manifest.parent_tree_hash,
        candidate_tree_hash=forged_tree.tree_hash,
        changes=tuple(
            ExpertCandidatePatchChange(
                relative_path=path,
                before=parent_files.get(path),
                after=candidate_files.get(path),
            )
            for path in sorted(set(parent_files) | set(candidate_files))
            if parent_files.get(path) != candidate_files.get(path)
        ),
    )
    materialization = closure.derivation.materialization
    forged_materialization = _remint(
        materialization,
        patch=forged_patch,
        source_tree=forged_tree,
        stable_authority_ids=tuple(
            sorted(
                {
                    materialization.composition_assessment.assessment_id,
                    *materialization.composition_assessment.stable_authority_ids,
                    materialization.parent_tree.source_tree_manifest_id,
                    forged_patch.patch_id,
                    forged_tree.source_tree_manifest_id,
                    materialization.repository_map.repository_map_id,
                    *(
                        module.module_contract_id
                        for module in materialization.module_contracts
                    ),
                }
            )
        ),
    )
    record = _remint(
        closure.derivation.record,
        composition_materialization_id=forged_materialization.materialization_id,
    )
    derivation = ExpertDeterministicCompositionDerivation(
        record=record,
        materialization=forged_materialization,
        source_provenance=closure.derivation.source_provenance,
        parent_contents=closure.derivation.parent_contents,
    )
    sanitation = reducer_case.source._resolver.candidate_store.validator.sanitizer.scan(
        closure.manifest.scope_contract_id,
        forged_tree,
        forged_contents,
    )
    manifest_payload = {
        field.name: getattr(closure.manifest, field.name)
        for field in fields(ExpertCandidateManifest)
        if field.name != ExpertCandidateManifest.IDENTITY_FIELD
    }
    manifest_payload.update(
        derivation_ref=record.derivation_id,
        patch_ref=forged_patch.patch_id,
        patch_digest=tree_or_blob_digest(forged_patch.to_json_bytes()),
        candidate_tree_ref=forged_tree.source_tree_manifest_id,
        candidate_tree_hash=forged_tree.tree_hash,
        sanitation_report_id=sanitation.sanitation_report_id,
    )
    forged = replace(
        closure,
        manifest=ExpertCandidateManifest.mint(**manifest_payload),
        patch=forged_patch,
        candidate_tree=forged_tree,
        derivation=derivation,
        sanitation_report=sanitation,
        candidate_contents=forged_contents,
    )

    with pytest.raises(ValueError, match="materialization or provenance"):
        reducer_case.source._resolver.candidate_store.validator.validate(forged)


def test_nonclean_composition_does_not_project_candidate(reducer_case):
    closure = reducer_case.source.stored_candidate.closure
    installed_base = _released_base(
        source=reducer_case.source,
        repository_map=closure.repository_map,
        module_contracts=closure.module_contracts,
        source_contents=closure.candidate_contents,
        label="composition candidate already installed",
    )
    plan = _plan(installed_base, (reducer_case.source,))
    reduction = reducer_case.reducer.reduce(
        plan=plan,
        current_base=installed_base,
        sources=(reducer_case.source.reduction_source,),
    )

    with pytest.raises(ExpertCompositionCandidateError, match="exact clean"):
        project_deterministic_composition_candidate(
            reduction=reduction,
            current_base=installed_base,
            approved_sources=(reducer_case.source,),
            sanitizer=reducer_case.source._resolver.candidate_store.validator.sanitizer,
        )
