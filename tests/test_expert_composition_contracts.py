from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

import pytest

from kapso.cross_run.canonical import content_id, tree_or_blob_digest
from kapso.cross_run.contracts import CandidateChangeKind
from kapso.cross_run.expert.composition_contracts import (
    ExpertCompositionAssessment,
    ExpertCompositionBaseReference,
    ExpertCompositionConflict,
    ExpertCompositionConflictKind,
    ExpertCompositionConflictSubjectKind,
    ExpertCompositionContractError,
    ExpertCompositionDisposition,
    ExpertCompositionPlan,
    ExpertCompositionSourceReference,
)
from test_expert_triggers import expert_records


def _id(namespace: str, label: str) -> str:
    return content_id(namespace, {"label": label})


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


def _base_reference(scope, module, repository_map, release):
    authorities = {
        release.release_id,
        scope.scope_contract_id,
        repository_map.repository_map_id,
        module.module_contract_id,
    }
    return ExpertCompositionBaseReference.mint(
        release_id=release.release_id,
        scope_contract_id=scope.scope_contract_id,
        scope_id=scope.scope_id,
        source_tree_hash=_digest("current source tree"),
        repository_map_id=repository_map.repository_map_id,
        module_contract_ids=(module.module_contract_id,),
        semantic_book_digest=release.semantic_book_digest,
        release_configuration_fingerprint=release.configuration_fingerprint,
        stable_authority_ids=tuple(sorted(authorities)),
    )


def _source_reference(scope, module, *, label):
    candidate_id = _id("expert-candidate", label)
    commit_id = _id("expert-candidate-commit", label)
    parent_release_id = _id("expert-base-release", f"{label} stale parent")
    parent_map_id = _id("expert-repository-map", f"{label} stale parent")
    patch_id = _id("expert-candidate-patch", label)
    proposed_map_id = _id("expert-repository-map", f"{label} proposed")
    authorities = {
        candidate_id,
        commit_id,
        scope.scope_contract_id,
        parent_release_id,
        parent_map_id,
        patch_id,
        proposed_map_id,
        module.module_contract_id,
    }
    return ExpertCompositionSourceReference.mint(
        candidate_id=candidate_id,
        candidate_commit_record_id=commit_id,
        scope_contract_id=scope.scope_contract_id,
        change_kind=CandidateChangeKind.CAPABILITY,
        parent_release_id=parent_release_id,
        parent_repository_map_id=parent_map_id,
        parent_tree_hash=_digest(f"{label} parent tree"),
        candidate_tree_hash=_digest(f"{label} candidate tree"),
        patch_id=patch_id,
        patch_digest=_digest(f"{label} patch"),
        proposed_repository_map_id=proposed_map_id,
        module_contract_ids=(module.module_contract_id,),
        candidate_configuration_fingerprint=_digest(f"{label} configuration"),
        stable_authority_ids=tuple(sorted(authorities)),
    )


def _plan_authorities(scope, base, sources):
    authorities = {
        scope.scope_contract_id,
        base.base_reference_id,
        *base.stable_authority_ids,
        *(source.source_reference_id for source in sources),
        *(
            authority_id
            for source in sources
            for authority_id in source.stable_authority_ids
        ),
    }
    if scope.supersedes_scope_contract_id is not None:
        authorities.add(scope.supersedes_scope_contract_id)
    return tuple(sorted(authorities))


def _plan(scope, base, sources):
    sources = tuple(
        sorted(
            sources,
            key=lambda source: (source.candidate_id, source.source_reference_id),
        )
    )
    return ExpertCompositionPlan.mint(
        scope_contract=scope,
        current_base=base,
        sources=sources,
        composition_policy_version="kapso.expert_composition.v1",
        configuration_fingerprint=_digest("composition configuration"),
        stable_authority_ids=_plan_authorities(scope, base, sources),
    )


def _conflict(kind, plan, source_indexes=None):
    subject_kind, subjects = {
        ExpertCompositionConflictKind.SOURCE_BASE: (
            ExpertCompositionConflictSubjectKind.AUTHORITY,
            (plan.current_base.release_id,),
        ),
        ExpertCompositionConflictKind.PATH_OVERLAP: (
            ExpertCompositionConflictSubjectKind.PATH,
            ("src/shared.py",),
        ),
        ExpertCompositionConflictKind.MODULE: (
            ExpertCompositionConflictSubjectKind.CAPABILITY,
            ("shared.capability",),
        ),
        ExpertCompositionConflictKind.TOPOLOGY: (
            ExpertCompositionConflictSubjectKind.CAPABILITY,
            ("shared.capability",),
        ),
        ExpertCompositionConflictKind.ADAPTER_BOUNDARY: (
            ExpertCompositionConflictSubjectKind.PATH,
            ("task_adapter",),
        ),
        ExpertCompositionConflictKind.ARCHITECTURE: (
            ExpertCompositionConflictSubjectKind.REPOSITORY,
            ("repository",),
        ),
    }[kind]
    if source_indexes is None:
        source_indexes = (
            (0, 1)
            if kind
            in {
                ExpertCompositionConflictKind.PATH_OVERLAP,
                ExpertCompositionConflictKind.PATH_PREFIX,
            }
            else (0,)
        )
    source_ids = tuple(
        sorted(plan.sources[index].source_reference_id for index in source_indexes)
    )
    return ExpertCompositionConflict.mint(
        kind=kind,
        subject_kind=subject_kind,
        subjects=subjects,
        source_reference_ids=source_ids,
    )


def _assessment(
    plan,
    disposition,
    *,
    applicable=(),
    already_present=(),
    conflicts=(),
):
    conflicts = tuple(sorted(conflicts, key=lambda conflict: conflict.canonical_key))
    authorities = {
        plan.composition_plan_id,
        *plan.stable_authority_ids,
        *(conflict.conflict_id for conflict in conflicts),
    }
    return ExpertCompositionAssessment.mint(
        composition_plan=plan,
        disposition=disposition,
        applicable_source_reference_ids=tuple(sorted(applicable)),
        already_present_source_reference_ids=tuple(sorted(already_present)),
        conflicts=conflicts,
        stable_authority_ids=tuple(sorted(authorities)),
    )


@pytest.fixture(scope="module")
def composition_case():
    scope, module, repository_map, release = expert_records()
    base = _base_reference(scope, module, repository_map, release)
    sources = tuple(
        _source_reference(scope, module, label=label) for label in ("first", "second")
    )
    plan = _plan(scope, base, sources)
    return SimpleNamespace(
        scope=scope,
        module=module,
        repository_map=repository_map,
        release=release,
        base=base,
        sources=plan.sources,
        plan=plan,
    )


def test_stable_references_plan_conflict_and_assessment_roundtrip(composition_case):
    plan = composition_case.plan
    first = plan.source_reference_ids[0]
    conflict = _conflict(ExpertCompositionConflictKind.PATH_OVERLAP, plan)
    assessment = _assessment(
        plan,
        ExpertCompositionDisposition.CONFLICTED,
        conflicts=(conflict,),
    )

    for record in (*plan.sources, plan.current_base, plan, conflict, assessment):
        assert type(record).from_json_bytes(record.to_json_bytes()) == record
    assert assessment.composition_plan == plan
    assert first in conflict.source_reference_ids


@pytest.mark.parametrize(
    ("record_name", "changes", "message"),
    (
        (
            "base",
            {"release_id": _id("wrong-release", "base")},
            "wrong namespace",
        ),
        (
            "source",
            {"candidate_id": _id("wrong-candidate", "source")},
            "wrong namespace",
        ),
        (
            "base",
            {"source_tree_hash": "not-a-digest"},
            "sha256 digest",
        ),
        (
            "source",
            {"patch_digest": "not-a-digest"},
            "sha256 digest",
        ),
    ),
)
def test_stable_references_reject_wrong_namespaces_and_digests(
    composition_case,
    record_name,
    changes,
    message,
):
    record = (
        composition_case.base if record_name == "base" else composition_case.sources[0]
    )

    with pytest.raises(ExpertCompositionContractError, match=message):
        _remint(record, **changes)


@pytest.mark.parametrize("record_name", ("base", "source", "plan"))
@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_stable_authority_closures_are_exact(
    composition_case,
    record_name,
    mutation,
):
    record = {
        "base": composition_case.base,
        "source": composition_case.sources[0],
        "plan": composition_case.plan,
    }[record_name]
    authorities = record.stable_authority_ids
    changed = (
        authorities[1:]
        if mutation == "missing"
        else tuple(sorted((*authorities, _id("unexpected-authority", record_name))))
    )

    with pytest.raises(ExpertCompositionContractError, match="closure is not exact"):
        _remint(record, stable_authority_ids=changed)


def test_plan_requires_canonical_unique_sources_in_one_scope(composition_case):
    plan = composition_case.plan
    first, second = plan.sources
    foreign_source = _remint(
        first,
        scope_contract_id=_id("expert-scope-contract", "foreign scope"),
        stable_authority_ids=tuple(
            sorted(
                {
                    *first.stable_authority_ids,
                    _id("expert-scope-contract", "foreign scope"),
                }
                - {first.scope_contract_id}
            )
        ),
    )

    with pytest.raises(ExpertCompositionContractError, match="canonical, and unique"):
        _remint(plan, sources=(second, first))
    with pytest.raises(ExpertCompositionContractError, match="canonical, and unique"):
        _remint(plan, sources=(first, first))
    with pytest.raises(ExpertCompositionContractError, match="one exact scope"):
        _plan(plan.scope_contract, plan.current_base, (foreign_source,))


def test_plan_deliberately_accepts_sources_from_stale_parents(
    composition_case,
):
    plan = composition_case.plan

    assert all(
        source.parent_release_id != plan.current_base.release_id
        and source.parent_repository_map_id != plan.current_base.repository_map_id
        and source.parent_tree_hash != plan.current_base.source_tree_hash
        for source in plan.sources
    )
    assert ExpertCompositionPlan.from_json_bytes(plan.to_json_bytes()) == plan


def test_plan_identity_contains_no_terminal_or_temporal_authority(
    composition_case,
):
    plan = composition_case.plan
    serialized_keys = set(plan.to_dict())

    assert serialized_keys.isdisjoint(
        {
            "current_release_observation",
            "publication_eligibility_result",
            "publication_authority_fence",
            "publisher_attestation",
            "security_denylist_observation",
            "validation_transition",
        }
    )
    assert {
        "scope_contract",
        "current_base",
        "sources",
        "stable_authority_ids",
    }.issubset(serialized_keys)


@pytest.mark.parametrize(
    "kind",
    (
        ExpertCompositionConflictKind.SOURCE_BASE,
        ExpertCompositionConflictKind.PATH_OVERLAP,
        ExpertCompositionConflictKind.MODULE,
        ExpertCompositionConflictKind.ARCHITECTURE,
    ),
)
def test_conflict_subject_syntax_is_typed_by_conflict_kind(
    composition_case,
    kind,
):
    conflict = _conflict(kind, composition_case.plan)

    assert (
        ExpertCompositionConflict.from_json_bytes(conflict.to_json_bytes()) == conflict
    )


@pytest.mark.parametrize(
    ("kind", "subject_kind", "subjects"),
    (
        (
            ExpertCompositionConflictKind.PATH_OVERLAP,
            ExpertCompositionConflictSubjectKind.CAPABILITY,
            ("shared.capability",),
        ),
        (
            ExpertCompositionConflictKind.PATH_OVERLAP,
            ExpertCompositionConflictSubjectKind.PATH,
            ("../escape.py",),
        ),
        (
            ExpertCompositionConflictKind.MODULE,
            ExpertCompositionConflictSubjectKind.CAPABILITY,
            ("",),
        ),
        (
            ExpertCompositionConflictKind.ARCHITECTURE,
            ExpertCompositionConflictSubjectKind.REPOSITORY,
            ("whole-repository",),
        ),
    ),
)
def test_conflict_rejects_kind_subject_mismatch_and_noncanonical_syntax(
    composition_case,
    kind,
    subject_kind,
    subjects,
):
    with pytest.raises(ValueError):
        ExpertCompositionConflict.mint(
            kind=kind,
            subject_kind=subject_kind,
            subjects=subjects,
            source_reference_ids=(composition_case.plan.source_reference_ids[0],),
        )


def test_assessment_requires_an_exact_disjoint_full_partition(composition_case):
    plan = composition_case.plan
    first, second = plan.source_reference_ids
    valid = _assessment(
        plan,
        ExpertCompositionDisposition.CLEAN,
        applicable=(first,),
        already_present=(second,),
    )

    assert valid.disposition is ExpertCompositionDisposition.CLEAN
    with pytest.raises(ExpertCompositionContractError, match="classifications"):
        _remint(
            valid,
            applicable_source_reference_ids=(first,),
            already_present_source_reference_ids=(first, second),
        )
    with pytest.raises(ExpertCompositionContractError, match="classify every"):
        _remint(
            valid,
            applicable_source_reference_ids=(first,),
            already_present_source_reference_ids=(),
        )


def test_assessment_rejects_architecture_source_as_mechanically_applicable(
    composition_case,
):
    plan = composition_case.plan
    architecture_source = _remint(
        plan.sources[0],
        change_kind=CandidateChangeKind.REPOSITORY_ARCHITECTURE,
    )
    architecture_plan = _plan(
        plan.scope_contract,
        plan.current_base,
        (architecture_source,),
    )

    with pytest.raises(ExpertCompositionContractError, match="only capability"):
        _assessment(
            architecture_plan,
            ExpertCompositionDisposition.CLEAN,
            applicable=architecture_plan.source_reference_ids,
        )

    path_conflict = ExpertCompositionConflict.mint(
        kind=ExpertCompositionConflictKind.CURRENT_PATH,
        subject_kind=ExpertCompositionConflictSubjectKind.PATH,
        subjects=("src/conflict.py",),
        source_reference_ids=architecture_plan.source_reference_ids,
    )
    with pytest.raises(
        ExpertCompositionContractError,
        match="architecture source requires",
    ):
        _assessment(
            architecture_plan,
            ExpertCompositionDisposition.CONFLICTED,
            conflicts=(path_conflict,),
        )

    architecture_conflict = _conflict(
        ExpertCompositionConflictKind.ARCHITECTURE,
        architecture_plan,
    )
    assessment = _assessment(
        architecture_plan,
        ExpertCompositionDisposition.REQUIRES_RESTRUCTURE,
        conflicts=(architecture_conflict,),
    )
    assert assessment.disposition is ExpertCompositionDisposition.REQUIRES_RESTRUCTURE


def test_cross_source_path_conflict_requires_multiple_sources(composition_case):
    plan = composition_case.plan

    with pytest.raises(ExpertCompositionContractError, match="multiple sources"):
        _conflict(
            ExpertCompositionConflictKind.PATH_OVERLAP,
            plan,
            source_indexes=(0,),
        )


def test_source_base_conflict_must_cite_planned_authority(composition_case):
    plan = composition_case.plan
    conflict = ExpertCompositionConflict.mint(
        kind=ExpertCompositionConflictKind.SOURCE_BASE,
        subject_kind=ExpertCompositionConflictSubjectKind.AUTHORITY,
        subjects=(_id("expert-base-release", "unrelated"),),
        source_reference_ids=(plan.source_reference_ids[0],),
    )

    with pytest.raises(ExpertCompositionContractError, match="outside its plan"):
        _assessment(
            plan,
            ExpertCompositionDisposition.CONFLICTED,
            already_present=(plan.source_reference_ids[1],),
            conflicts=(conflict,),
        )


@pytest.mark.parametrize(
    ("kind", "valid_disposition", "invalid_disposition"),
    (
        (
            ExpertCompositionConflictKind.PATH_OVERLAP,
            ExpertCompositionDisposition.CONFLICTED,
            ExpertCompositionDisposition.REQUIRES_RESTRUCTURE,
        ),
        (
            ExpertCompositionConflictKind.TOPOLOGY,
            ExpertCompositionDisposition.REQUIRES_RESTRUCTURE,
            ExpertCompositionDisposition.CONFLICTED,
        ),
        (
            ExpertCompositionConflictKind.ADAPTER_BOUNDARY,
            ExpertCompositionDisposition.REQUIRES_RESTRUCTURE,
            ExpertCompositionDisposition.CONFLICTED,
        ),
    ),
)
def test_conflict_kind_deterministically_selects_conflict_disposition(
    composition_case,
    kind,
    valid_disposition,
    invalid_disposition,
):
    plan = composition_case.plan
    conflict = _conflict(kind, plan)
    valid = _assessment(
        plan,
        valid_disposition,
        already_present=tuple(
            sorted(set(plan.source_reference_ids) - set(conflict.source_reference_ids))
        ),
        conflicts=(conflict,),
    )

    assert valid.disposition is valid_disposition
    with pytest.raises(ExpertCompositionContractError, match="disposition"):
        _remint(valid, disposition=invalid_disposition)


def test_already_present_requires_the_entire_plan_partition(composition_case):
    plan = composition_case.plan
    first, second = plan.source_reference_ids

    complete = _assessment(
        plan,
        ExpertCompositionDisposition.ALREADY_PRESENT,
        already_present=(first, second),
    )
    assert complete.already_present_source_reference_ids == (first, second)
    with pytest.raises(ExpertCompositionContractError, match="disposition"):
        _assessment(
            plan,
            ExpertCompositionDisposition.ALREADY_PRESENT,
            applicable=(first,),
            already_present=(second,),
        )


@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_assessment_stable_authority_closure_is_exact(composition_case, mutation):
    plan = composition_case.plan
    conflict = _conflict(ExpertCompositionConflictKind.PATH_OVERLAP, plan)
    assessment = _assessment(
        plan,
        ExpertCompositionDisposition.CONFLICTED,
        conflicts=(conflict,),
    )
    authorities = assessment.stable_authority_ids
    changed = (
        tuple(value for value in authorities if value != conflict.conflict_id)
        if mutation == "missing"
        else tuple(sorted((*authorities, _id("unexpected-authority", "assessment"))))
    )

    with pytest.raises(ExpertCompositionContractError, match="closure is not exact"):
        _remint(assessment, stable_authority_ids=changed)
