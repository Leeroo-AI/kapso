"""Authenticated author boundary for expert release-use revocations."""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

import kapso.cross_run.catalog.release_use_authority as release_use_authority_module
from kapso.cross_run.canonical import content_id
from kapso.cross_run.catalog.release_use_authority import (
    CatalogReleaseUseRevocationAuthority,
    CatalogReleaseUseRevocationAuthorityError,
    _seal_catalog_release_use_revocation_authority,
)
from kapso.cross_run.catalog.service import CrossRunCatalogError
from kapso.cross_run.catalog.store import (
    CatalogCompareAndSwapError,
    CatalogInputDelta,
)
from kapso.cross_run.expert.release_authority import (
    ExpertReleaseActivationAuthorityError,
)
from kapso.cross_run.expert.release_use import (
    ExpertReleaseUseRevocationAuthor,
    PendingExpertReleaseUseRevocation,
)
from kapso.cross_run.record_contracts import (
    ExpertReleaseUseRevocation,
    ExpertReleaseUseRevocationKind,
)
from test_expert_release_authority import _authority_fixture
from test_expert_release_use_catalog import _external_id, _projected_catalog

_RECORDED_AT = "2026-07-23T00:00:00Z"


def _author_case(tmp_path: Path, *, witness_overrides=()):
    catalog, projected, evidence_id = _projected_catalog(tmp_path)
    case, _, _, _, resolver, _, provider = _authority_fixture(
        witness_overrides=witness_overrides,
    )
    assert case.scope == catalog.scope_contract
    author = ExpertReleaseUseRevocationAuthor(catalog, provider)
    return catalog, projected, evidence_id, case, resolver, provider, author


def _publish(
    author: ExpertReleaseUseRevocationAuthor,
    *,
    expected_generation,
    release_id: str,
    evidence_id: str,
    kind: ExpertReleaseUseRevocationKind = (ExpertReleaseUseRevocationKind.PERFORMANCE),
    recorded_at: str = _RECORDED_AT,
) -> PendingExpertReleaseUseRevocation:
    return author.publish(
        expected_generation=expected_generation,
        release_id=release_id,
        kind=kind,
        reason_code=f"{kind.value}_regression",
        rationale=f"Independent evidence found a {kind.value} regression.",
        exact_evidence_refs=(evidence_id,),
        recorded_at=recorded_at,
    )


def test_author_publishes_exact_delta_and_replays_against_fresh_current(
    tmp_path: Path,
) -> None:
    catalog, projected, evidence_id, case, _, _, author = _author_case(tmp_path)

    first = _publish(
        author,
        expected_generation=projected,
        release_id=case.release.release_id,
        evidence_id=evidence_id,
    )
    committed = first.catalog_commit.generation
    new_delta_ids = set(committed.applied_input_delta_ids) - set(
        projected.applied_input_delta_ids
    )
    assert len(new_delta_ids) == 1
    delta = catalog.store.read_contract(new_delta_ids.pop(), CatalogInputDelta)
    assert delta.operation_id == content_id(
        "expert-release-use-revocation-operation",
        {"revocation_id": first.event.revocation_id},
    )
    assert delta.added_object_ids == (first.event.revocation_id,)
    assert delta.dependency_closure_ids == tuple(
        sorted((evidence_id, first.event.revocation_id))
    )
    assert first.event.release_publication_id not in committed.fact_object_ids
    assert first.event.release_activation_witness_id not in committed.fact_object_ids
    assert first.event.revocation_id not in committed.active_entry_state_ids

    replay = _publish(
        author,
        expected_generation=committed,
        release_id=case.release.release_id,
        evidence_id=evidence_id,
    )

    assert replay.event == first.event
    assert replay.catalog_commit.replayed
    assert replay.catalog_commit.generation == committed
    assert catalog.store.read_current() == committed


def test_distinct_release_use_findings_accumulate_without_scientific_taint(
    tmp_path: Path,
) -> None:
    catalog, projected, evidence_id, case, _, _, author = _author_case(tmp_path)
    performance = _publish(
        author,
        expected_generation=projected,
        release_id=case.release.release_id,
        evidence_id=evidence_id,
    )
    compatibility = _publish(
        author,
        expected_generation=performance.catalog_commit.generation,
        release_id=case.release.release_id,
        evidence_id=evidence_id,
        kind=ExpertReleaseUseRevocationKind.COMPATIBILITY,
    )

    view = catalog.read_generation(compatibility.catalog_commit.generation)

    assert view.facts.release_use_revocations == tuple(
        sorted(
            (performance.event, compatibility.event),
            key=lambda event: event.revocation_id,
        )
    )
    assert all(
        not state.revocation_ids and not state.taint_source_ids
        for state in view.entry_states
    )


@pytest.mark.parametrize("invalid_request", ("missing_evidence", "stale_generation"))
def test_invalid_local_generation_fails_before_provider_resolution(
    tmp_path: Path,
    invalid_request: str,
) -> None:
    catalog, projected, evidence_id, case, resolver, _, author = _author_case(tmp_path)
    selected_evidence = evidence_id
    if invalid_request == "missing_evidence":
        selected_evidence = _external_id("run-bundle", "missing")
        message = "evidence is absent"
    else:
        catalog.rebase(
            operation_id="advance_before_release_use_request",
            objects=(),
            dependency_closure_ids=(),
        )
        message = "no longer current"

    with pytest.raises(CrossRunCatalogError, match=message):
        _publish(
            author,
            expected_generation=projected,
            release_id=case.release.release_id,
            evidence_id=selected_evidence,
        )

    assert resolver.resolve_calls == []


def test_provider_failure_leaves_catalog_unchanged(tmp_path: Path) -> None:
    catalog, projected, evidence_id, case, resolver, _, author = _author_case(
        tmp_path,
        witness_overrides=(None,),
    )

    with pytest.raises(
        ExpertReleaseActivationAuthorityError,
        match="witness is missing",
    ):
        _publish(
            author,
            expected_generation=projected,
            release_id=case.release.release_id,
            evidence_id=evidence_id,
        )

    assert resolver.resolve_calls
    assert catalog.store.read_current() == projected
    assert not catalog.read_current().facts.release_use_revocations


def test_remote_validation_race_fails_cas_without_implicit_rebase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog, projected, evidence_id, case, resolver, _, author = _author_case(tmp_path)
    resolve_artifact = resolver.resolve_artifact

    def resolve_with_race(scope_id, artifact_kind, release_id):
        if len(resolver.resolve_calls) == 2:
            catalog.rebase(
                operation_id="advance_during_release_use_authentication",
                objects=(),
                dependency_closure_ids=(),
            )
        return resolve_artifact(scope_id, artifact_kind, release_id)

    monkeypatch.setattr(resolver, "resolve_artifact", resolve_with_race)

    with pytest.raises(CatalogCompareAndSwapError):
        _publish(
            author,
            expected_generation=projected,
            release_id=case.release.release_id,
            evidence_id=evidence_id,
        )

    current = catalog.read_current()
    assert current.generation.generation_number == projected.generation_number + 1
    assert not current.facts.release_use_revocations


def test_capability_is_sealed_catalog_bound_and_single_author(
    tmp_path: Path,
) -> None:
    catalog, _, _, _, _, provider, author = _author_case(tmp_path)

    with pytest.raises(
        CatalogReleaseUseRevocationAuthorityError,
        match="not author sealed",
    ):
        CatalogReleaseUseRevocationAuthority(
            object(),
            author=author,
            catalog=catalog,
        )
    with pytest.raises(
        CatalogReleaseUseRevocationAuthorityError,
        match="cannot be serialized",
    ):
        pickle.dumps(author._authority)
    foreign = _seal_catalog_release_use_revocation_authority(
        author=author,
        catalog=object(),
    )
    with pytest.raises(CrossRunCatalogError, match="foreign authority"):
        catalog._publish_authenticated_release_use_revocation(
            authority=foreign,
            historical_activation=object(),
            expected_generation=catalog.store.read_current(),
            event=object(),
        )
    with pytest.raises(CrossRunCatalogError, match="already has another"):
        ExpertReleaseUseRevocationAuthor(catalog, provider)


def test_foreign_process_author_fails_before_historical_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog, projected, evidence_id, case, resolver, _, author = _author_case(tmp_path)
    owner_process_id = author._authority._owner_process_id
    monkeypatch.setattr(
        release_use_authority_module.os,
        "getpid",
        lambda: owner_process_id + 1,
    )

    with pytest.raises(
        CatalogReleaseUseRevocationAuthorityError,
        match="authority is foreign",
    ):
        _publish(
            author,
            expected_generation=projected,
            release_id=case.release.release_id,
            evidence_id=evidence_id,
        )

    assert resolver.resolve_calls == []
    assert catalog.store.read_current() == projected

def test_authenticated_path_rejects_event_that_does_not_join_activation(
    tmp_path: Path,
) -> None:
    catalog, projected, evidence_id, case, _, provider, author = _author_case(tmp_path)
    activation = provider.resolve_exact(
        catalog.scope_contract,
        case.release.release_id,
    )
    event = ExpertReleaseUseRevocation.mint(
        scope_contract_id=catalog.scope_contract.scope_contract_id,
        scope_id=catalog.scope_contract.scope_id,
        release_id=activation.manifest.release_id,
        release_publication_id=_external_id("github-publication", "forged"),
        release_activation_witness_id=activation.witness.witness_id,
        kind=ExpertReleaseUseRevocationKind.PERFORMANCE,
        reason_code="performance_regression",
        rationale="Independent evidence found a performance regression.",
        exact_evidence_refs=(evidence_id,),
        recorded_at=_RECORDED_AT,
    )

    with pytest.raises(
        ValueError,
        match="does not join historical activation exactly",
    ):
        catalog._publish_authenticated_release_use_revocation(
            authority=author._authority,
            historical_activation=activation,
            expected_generation=projected,
            event=event,
        )

    assert catalog.store.read_current() == projected
