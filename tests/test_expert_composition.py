from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

import pytest

from kapso.cross_run.canonical import content_id, tree_or_blob_digest
from kapso.cross_run.contracts import (
    ExpertCandidatePatchChange,
    ExpertReleaseLineage,
    SourceFileDescriptor,
)
from kapso.cross_run.expert.composition import (
    ExpertCompositionError,
    ExpertCompositionReducer,
    _ModuleEffect,
    _SourceEffect,
)
from kapso.cross_run.expert.composition_base import (
    build_expert_composition_base_closure,
)
from kapso.cross_run.expert.composition_contracts import (
    ExpertCompositionConflictKind,
    ExpertCompositionDisposition,
    ExpertCompositionPlan,
    expert_composition_configuration_fingerprint,
)
from kapso.cross_run.expert.composition_source import (
    ExpertCompositionSourceResolver,
)
from kapso.cross_run.expert.book import expert_control_paths
from test_expert_composition_base import _parent_receipt
from test_expert_composition_source import _terminalize
from test_expert_publication_eligibility import terminal_cases
from test_expert_triggers import expert_records


def _digest(label: str) -> str:
    return tree_or_blob_digest(label.encode("utf-8"))


def _remint(record, **changes):
    payload = {
        field.name: getattr(record, field.name)
        for field in fields(record)
        if field.name != record.IDENTITY_FIELD
    }
    payload.update(changes)
    return type(record).mint(**payload)


def _plan(current_base, sources):
    source_references = tuple(
        sorted(
            (source.source_reference for source in sources),
            key=lambda reference: (
                reference.candidate_id,
                reference.source_reference_id,
            ),
        )
    )
    authorities = {
        current_base.scope_contract.scope_contract_id,
        current_base.reference.base_reference_id,
        *current_base.reference.stable_authority_ids,
        *(reference.source_reference_id for reference in source_references),
        *(
            authority_id
            for reference in source_references
            for authority_id in reference.stable_authority_ids
        ),
    }
    superseded = current_base.scope_contract.supersedes_scope_contract_id
    if superseded is not None:
        authorities.add(superseded)
    expert_settings = sources[0]._resolver.candidate_store.validator.settings
    return ExpertCompositionPlan.mint(
        scope_contract=current_base.scope_contract,
        current_base=current_base.reference,
        sources=source_references,
        active_task_bindings=(
            sources[0].stored_candidate.closure.validation_context.active_task_bindings
        ),
        composition_policy_version=expert_settings.composition_policy_version,
        composition_source_limit=expert_settings.composition_source_limit,
        candidate_entry_limit=expert_settings.candidate_entry_limit,
        candidate_byte_limit=expert_settings.candidate_byte_limit,
        configuration_fingerprint=expert_composition_configuration_fingerprint(
            composition_policy_version=expert_settings.composition_policy_version,
            composition_source_limit=expert_settings.composition_source_limit,
            candidate_entry_limit=expert_settings.candidate_entry_limit,
            candidate_byte_limit=expert_settings.candidate_byte_limit,
        ),
        stable_authority_ids=tuple(sorted(authorities)),
    )


def _released_base(
    *,
    source,
    repository_map,
    module_contracts,
    source_contents,
    label,
):
    closure = source.stored_candidate.closure
    source_base_release = closure.derivation.trigger_packet.source_base_release
    if source_base_release is None:
        raise AssertionError("composition test source unexpectedly bootstraps")
    book = source_contents["EXPERT_REPO.md"]
    dependency_closure = set(source_base_release.consumed_dependency_ids)
    dependency_closure.add(repository_map.repository_map_id)
    dependency_closure.add(source_base_release.release_id)
    dependency_closure.update(module.module_contract_id for module in module_contracts)
    release = _remint(
        source_base_release,
        lineage=ExpertReleaseLineage(
            source_base_release_id=source_base_release.release_id,
            activation_predecessor_release_id=source_base_release.release_id,
        ),
        repository_map_ref=repository_map.repository_map_id,
        module_contract_refs=tuple(
            sorted(module.module_contract_id for module in module_contracts)
        ),
        module_versions={
            module.module_id: module.version for module in module_contracts
        },
        semantic_book_digest=tree_or_blob_digest(book),
        consumed_dependency_ids=tuple(sorted(dependency_closure)),
        checksums={
            **source_base_release.checksums,
            source_base_release.source_archive_ref: _digest(f"{label} source archive"),
        },
    )
    receipt = _parent_receipt(
        release,
        repository_map,
        module_contracts,
        source_contents,
        cache_label=label,
    )
    return build_expert_composition_base_closure(
        scope_contract=closure.derivation.trigger_packet.scope_contract,
        release_manifest=release,
        source_base_tree_receipt=receipt,
        repository_map=repository_map,
        module_contracts=module_contracts,
        source_contents=source_contents,
    )


@pytest.fixture(scope="module")
def reducer_case(terminal_cases):
    terminal = _terminalize(terminal_cases.parent_approved)
    resolver = ExpertCompositionSourceResolver(
        terminal_cases.parent_approved.validation_store
    )
    source = resolver.resolve(terminal.snapshot.state.candidate_id)
    prepared_parent = terminal_cases.parent_approved.prepared.source_base
    if prepared_parent is None:
        raise AssertionError("composition test source lacks its released source base")
    closure = source.stored_candidate.closure
    source_base_release = _remint(
        prepared_parent.release_manifest,
        semantic_book_digest=tree_or_blob_digest(
            prepared_parent.source_contents["EXPERT_REPO.md"]
        ),
    )
    source_base_receipt = _parent_receipt(
        source_base_release,
        closure.derivation.trigger_packet.source_base_repository_map,
        closure.derivation.trigger_packet.source_base_module_contracts,
        prepared_parent.source_contents,
        cache_label="composition parent",
    )
    parent_base = build_expert_composition_base_closure(
        scope_contract=closure.derivation.trigger_packet.scope_contract,
        release_manifest=source_base_release,
        source_base_tree_receipt=source_base_receipt,
        repository_map=closure.derivation.trigger_packet.source_base_repository_map,
        module_contracts=closure.derivation.trigger_packet.source_base_module_contracts,
        source_contents=prepared_parent.source_contents,
    )
    reducer = ExpertCompositionReducer(
        candidate_entry_limit=len(closure.candidate_tree.files) + 10,
        candidate_byte_limit=(
            sum(descriptor.size for descriptor in closure.candidate_tree.files) + 10_000
        ),
    )
    return SimpleNamespace(
        source=source,
        parent_base=parent_base,
        reducer=reducer,
    )


def test_clean_reduction_recreates_exact_candidate_without_merging_controls(
    reducer_case,
):
    case = reducer_case
    plan = _plan(case.parent_base, (case.source,))

    reduction = case.reducer.reduce(
        plan=plan,
        current_base=case.parent_base,
        sources=(case.source.reduction_source,),
    )
    closure = case.source.stored_candidate.closure

    assert reduction.assessment.disposition is ExpertCompositionDisposition.CLEAN
    assert reduction.materialization is not None
    assert reduction.materialization.source_tree == closure.candidate_tree
    assert reduction.materialization.patch == closure.patch
    assert reduction.materialization.repository_map == closure.repository_map
    assert reduction.materialization.module_contracts == closure.module_contracts
    assert dict(reduction.source_contents) == dict(closure.candidate_contents)


def test_exact_installed_candidate_is_already_present_and_has_no_materialization(
    reducer_case,
):
    case = reducer_case
    closure = case.source.stored_candidate.closure
    installed_base = _released_base(
        source=case.source,
        repository_map=closure.repository_map,
        module_contracts=closure.module_contracts,
        source_contents=closure.candidate_contents,
        label="installed candidate",
    )
    plan = _plan(installed_base, (case.source,))

    reduction = case.reducer.reduce(
        plan=plan,
        current_base=installed_base,
        sources=(case.source.reduction_source,),
    )

    assert (
        reduction.assessment.disposition is ExpertCompositionDisposition.ALREADY_PRESENT
    )
    assert reduction.materialization is None
    assert not reduction.source_contents


def test_partially_present_source_applies_only_missing_module_effect(reducer_case):
    case = reducer_case
    closure = case.source.stored_candidate.closure
    controls = set(
        expert_control_paths(closure.derivation.trigger_packet.source_base_module_contracts)
    )
    controls.update(expert_control_paths(closure.module_contracts))
    partially_installed_contents = dict(case.parent_base.source_contents)
    for change in closure.patch.changes:
        if change.relative_path in controls:
            continue
        if change.after is None:
            partially_installed_contents.pop(change.relative_path)
        else:
            partially_installed_contents[change.relative_path] = (
                closure.candidate_contents[change.relative_path]
            )
    partially_installed_base = _released_base(
        source=case.source,
        repository_map=case.parent_base.repository_map,
        module_contracts=case.parent_base.module_contracts,
        source_contents=partially_installed_contents,
        label="partially installed candidate",
    )
    plan = _plan(partially_installed_base, (case.source,))

    reduction = case.reducer.reduce(
        plan=plan,
        current_base=partially_installed_base,
        sources=(case.source.reduction_source,),
    )

    assert reduction.assessment.disposition is ExpertCompositionDisposition.CLEAN
    assert reduction.materialization is not None
    assert reduction.materialization.source_tree == closure.candidate_tree
    assert reduction.materialization.module_contracts == closure.module_contracts
    assert dict(reduction.source_contents) == dict(closure.candidate_contents)


def test_third_current_path_value_conflicts_without_partial_materialization(
    reducer_case,
):
    case = reducer_case
    closure = case.source.stored_candidate.closure
    source_base_controls = {
        "EXPERT_REPO.md",
        ".kapso/expert/repository-map.json",
    }
    changed_path = next(
        change.relative_path
        for change in closure.patch.changes
        if change.relative_path not in source_base_controls
        and not change.relative_path.startswith(".kapso/expert/module-contracts/")
    )
    divergent_contents = dict(case.parent_base.source_contents)
    divergent_contents[changed_path] = b"independent current edit"
    divergent_base = _released_base(
        source=case.source,
        repository_map=case.parent_base.repository_map,
        module_contracts=case.parent_base.module_contracts,
        source_contents=divergent_contents,
        label="divergent current",
    )
    plan = _plan(divergent_base, (case.source,))

    reduction = case.reducer.reduce(
        plan=plan,
        current_base=divergent_base,
        sources=(case.source.reduction_source,),
    )

    assert reduction.assessment.disposition is ExpertCompositionDisposition.CONFLICTED
    assert {conflict.kind for conflict in reduction.assessment.conflicts} == {
        ExpertCompositionConflictKind.CURRENT_PATH
    }
    assert reduction.materialization is None
    assert not reduction.source_contents


def test_aggregate_candidate_limit_is_a_closed_conflict(reducer_case):
    case = reducer_case
    plan = _plan(case.parent_base, (case.source,))
    reducer = ExpertCompositionReducer(
        candidate_entry_limit=len(
            case.source.stored_candidate.closure.candidate_tree.files
        )
        - 1,
        candidate_byte_limit=sum(
            descriptor.size
            for descriptor in case.source.stored_candidate.closure.candidate_tree.files
        ),
    )

    reduction = reducer.reduce(
        plan=plan,
        current_base=case.parent_base,
        sources=(case.source.reduction_source,),
    )

    assert reduction.assessment.disposition is ExpertCompositionDisposition.CONFLICTED
    assert reduction.assessment.conflicts[0].kind is (
        ExpertCompositionConflictKind.RESOURCE_ENVELOPE
    )
    assert reduction.materialization is None


def test_runtime_closure_must_equal_the_plan(reducer_case):
    case = reducer_case
    closure = case.source.stored_candidate.closure
    installed_base = _released_base(
        source=case.source,
        repository_map=closure.repository_map,
        module_contracts=closure.module_contracts,
        source_contents=closure.candidate_contents,
        label="other plan base",
    )
    plan = _plan(installed_base, (case.source,))

    with pytest.raises(ExpertCompositionError, match="base differs"):
        case.reducer.reduce(
            plan=plan,
            current_base=case.parent_base,
            sources=(case.source.reduction_source,),
        )


def test_shared_already_path_still_conflicts_when_both_sources_are_applicable(
    reducer_case,
):
    reducer = reducer_case.reducer
    first_reference_id = content_id(
        "expert-composition-source-reference",
        {"source": "first"},
    )
    second_reference_id = content_id(
        "expert-composition-source-reference",
        {"source": "second"},
    )
    before = SourceFileDescriptor(
        relative_path="src/shared.py",
        digest=_digest("before"),
        mode="100644",
        size=len(b"before"),
    )
    after = SourceFileDescriptor(
        relative_path="src/shared.py",
        digest=_digest("after"),
        mode="100644",
        size=len(b"after"),
    )
    shared_change = ExpertCandidatePatchChange(
        relative_path="src/shared.py",
        before=before,
        after=after,
    )
    effects = (
        _SourceEffect(
            source=SimpleNamespace(
                source_reference=SimpleNamespace(source_reference_id=first_reference_id)
            ),
            editable_changes=(shared_change,),
            module_effects=(),
            already_present=False,
            individual_conflicts=(),
        ),
        _SourceEffect(
            source=SimpleNamespace(
                source_reference=SimpleNamespace(
                    source_reference_id=second_reference_id
                )
            ),
            editable_changes=(shared_change,),
            module_effects=(),
            already_present=False,
            individual_conflicts=(),
        ),
    )

    conflicts = reducer._cross_source_conflicts(effects)

    assert len(conflicts) == 1
    assert conflicts[0].kind is ExpertCompositionConflictKind.PATH_OVERLAP
    assert set(conflicts[0].source_reference_ids) == {
        first_reference_id,
        second_reference_id,
    }


def test_composed_incompatible_capabilities_cite_exact_contributing_sources():
    reducer = ExpertCompositionReducer(
        candidate_entry_limit=1,
        candidate_byte_limit=1,
    )
    _, original, _, _ = expert_records()
    other_module_id = "shared.other"
    first_after = _remint(
        original,
        incompatible_capability_ids=(other_module_id,),
    )
    other_before = _remint(
        original,
        module_id=other_module_id,
        incompatible_capability_ids=(),
    )
    other_after = _remint(
        other_before,
        incompatible_capability_ids=(original.module_id,),
    )
    first_reference_id = content_id(
        "expert-composition-source-reference",
        {"incompatible_source": "first"},
    )
    second_reference_id = content_id(
        "expert-composition-source-reference",
        {"incompatible_source": "second"},
    )
    effects = (
        _SourceEffect(
            source=SimpleNamespace(
                source_reference=SimpleNamespace(source_reference_id=first_reference_id)
            ),
            editable_changes=(),
            module_effects=(
                _ModuleEffect(
                    module_id=original.module_id,
                    before=original,
                    after=first_after,
                ),
            ),
            already_present=False,
            individual_conflicts=(),
        ),
        _SourceEffect(
            source=SimpleNamespace(
                source_reference=SimpleNamespace(
                    source_reference_id=second_reference_id
                )
            ),
            editable_changes=(),
            module_effects=(
                _ModuleEffect(
                    module_id=other_module_id,
                    before=other_before,
                    after=other_after,
                ),
            ),
            already_present=False,
            individual_conflicts=(),
        ),
    )

    conflicts = reducer._capability_incompatibility_conflicts(
        module_contracts=(first_after, other_after),
        effects=effects,
    )

    assert len(conflicts) == 1
    assert conflicts[0].kind is (
        ExpertCompositionConflictKind.CAPABILITY_INCOMPATIBILITY
    )
    assert conflicts[0].subjects == tuple(sorted((original.module_id, other_module_id)))
    assert set(conflicts[0].source_reference_ids) == {
        first_reference_id,
        second_reference_id,
    }

    one_sided_effect = reducer._capability_incompatibility_conflicts(
        module_contracts=(first_after, other_after),
        effects=(effects[0],),
    )
    same_source_unit = _SourceEffect(
        source=effects[0].source,
        editable_changes=(),
        module_effects=(effects[0].module_effects[0], effects[1].module_effects[0]),
        already_present=False,
        individual_conflicts=(),
    )
    approved_unit_effect = reducer._capability_incompatibility_conflicts(
        module_contracts=(first_after, other_after),
        effects=(same_source_unit,),
    )

    assert one_sided_effect == ()
    assert approved_unit_effect == ()
