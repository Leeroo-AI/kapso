"""Catalog boundaries for non-emergency expert release-use revocations."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from kapso.cross_run.canonical import CanonicalizationError, content_id
from kapso.cross_run.catalog.reducer import CatalogFactError
from kapso.cross_run.catalog.service import CrossRunCatalog
from kapso.cross_run.catalog.store import CatalogGenerationManifest
from kapso.cross_run.contracts import (
    ContractValidationError,
    MissingReferenceError,
    TransferEpisode,
)
from kapso.cross_run.record_contracts import (
    BundleProjectionManifest,
    CatalogRevocation,
    CatalogTaint,
    ExpertReleaseUseRevocation,
    ExpertReleaseUseRevocationKind,
)
from kapso.cross_run.record_registry import (
    CATALOG_FACT_RECORD_TYPES,
    KNOWLEDGE_RECORD_TYPES,
    parse_knowledge_record_payload,
)
from test_cross_run_catalog_service import _project_real_bundle, _scope_contract


def _external_id(namespace: str, label: str) -> str:
    return content_id(namespace, {"label": label})


def _revocation(
    *,
    evidence_id: str,
    kind: ExpertReleaseUseRevocationKind = (ExpertReleaseUseRevocationKind.PERFORMANCE),
    scope_contract_id: str | None = None,
    scope_id: str = "ml_ai",
    label: str = "finding",
) -> ExpertReleaseUseRevocation:
    scope = _scope_contract()
    return ExpertReleaseUseRevocation.mint(
        scope_contract_id=scope_contract_id or scope.scope_contract_id,
        scope_id=scope_id,
        release_id=_external_id("expert-base-release", label),
        release_publication_id=_external_id("github-publication", label),
        release_activation_witness_id=_external_id(
            "github-artifact-activation-witness",
            label,
        ),
        kind=kind,
        reason_code=f"{kind.value}_regression",
        rationale=f"Independent evidence found a {kind.value} regression.",
        exact_evidence_refs=(evidence_id,),
        recorded_at="2026-07-23T00:00:00Z",
    )


def _projected_catalog(
    tmp_path: Path,
) -> tuple[CrossRunCatalog, CatalogGenerationManifest, str]:
    fixture, projection = _project_real_bundle(tmp_path)
    catalog = CrossRunCatalog(
        tmp_path / "catalog",
        _scope_contract(),
        fixture.settings.catalog,
    )
    projected = catalog.publish_projection(
        catalog.store.read_current(),
        projection,
    ).generation
    return catalog, projected, projection.source_bundle.bundle_id


def test_release_use_revocation_is_a_strict_registered_catalog_fact() -> None:
    event = _revocation(evidence_id=_external_id("run-bundle", "evidence"))

    assert event.kind is ExpertReleaseUseRevocationKind.PERFORMANCE
    assert (
        CATALOG_FACT_RECORD_TYPES["expert-release-use-revocation"]
        is ExpertReleaseUseRevocation
    )
    assert (
        KNOWLEDGE_RECORD_TYPES["expert-release-use-revocation"]
        is ExpertReleaseUseRevocation
    )
    assert (
        parse_knowledge_record_payload(
            "expert-release-use-revocation",
            event.to_dict(),
        )
        == event
    )
    legacy_payload = event.to_dict()
    legacy_payload["activation_receipt_id"] = legacy_payload.pop(
        "release_activation_witness_id"
    )
    with pytest.raises(ContractValidationError, match="fields mismatch"):
        parse_knowledge_record_payload(
            "expert-release-use-revocation",
            legacy_payload,
        )

    with pytest.raises(ContractValidationError, match="must be one of"):
        ExpertReleaseUseRevocation.mint(
            scope_contract_id=event.scope_contract_id,
            scope_id=event.scope_id,
            release_id=event.release_id,
            release_publication_id=event.release_publication_id,
            release_activation_witness_id=event.release_activation_witness_id,
            kind="security",
            reason_code=event.reason_code,
            rationale=event.rationale,
            exact_evidence_refs=event.exact_evidence_refs,
            recorded_at=event.recorded_at,
        )
    with pytest.raises(ContractValidationError, match="wrong namespace"):
        replace(
            event,
            release_publication_id=_external_id("expert-base-release", "wrong"),
        )
    with pytest.raises(ContractValidationError, match="non-empty"):
        ExpertReleaseUseRevocation.mint(
            scope_contract_id=event.scope_contract_id,
            scope_id=event.scope_id,
            release_id=event.release_id,
            release_publication_id=event.release_publication_id,
            release_activation_witness_id=event.release_activation_witness_id,
            kind=event.kind,
            reason_code=event.reason_code,
            rationale=event.rationale,
            exact_evidence_refs=(),
            recorded_at=event.recorded_at,
        )
    second_evidence_id = _external_id("run-bundle", "second-evidence")
    with pytest.raises(ContractValidationError, match="sorted"):
        ExpertReleaseUseRevocation.mint(
            scope_contract_id=event.scope_contract_id,
            scope_id=event.scope_id,
            release_id=event.release_id,
            release_publication_id=event.release_publication_id,
            release_activation_witness_id=event.release_activation_witness_id,
            kind=event.kind,
            reason_code=event.reason_code,
            rationale=event.rationale,
            exact_evidence_refs=tuple(
                sorted((event.exact_evidence_refs[0], second_evidence_id), reverse=True)
            ),
            recorded_at=event.recorded_at,
        )
    with pytest.raises(CanonicalizationError, match="ISO-8601 UTC timestamp"):
        replace(event, recorded_at="not-a-timestamp")


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "message"),
    (
        (
            "scope_contract_id",
            _external_id("run-bundle", "wrong-scope-contract"),
            "wrong namespace",
        ),
        (
            "release_id",
            _external_id("run-bundle", "wrong-release"),
            "wrong namespace",
        ),
        (
            "release_activation_witness_id",
            _external_id("run-bundle", "wrong-activation"),
            "wrong namespace",
        ),
        ("scope_id", "not a valid identifier", "identifier"),
        ("reason_code", "not a valid identifier", "identifier"),
        ("rationale", " ", "must not be empty"),
    ),
)
def test_release_use_revocation_rejects_invalid_contract_fields(
    field_name: str,
    invalid_value: str,
    message: str,
) -> None:
    event = _revocation(evidence_id=_external_id("run-bundle", "evidence"))

    with pytest.raises(ValueError, match=message):
        replace(event, **{field_name: invalid_value})


def test_catalog_accumulates_release_use_events_without_admission_or_taint(
    tmp_path: Path,
) -> None:
    catalog, projected, evidence_id = _projected_catalog(tmp_path)
    performance = _revocation(evidence_id=evidence_id, label="performance")
    compatibility = _revocation(
        evidence_id=evidence_id,
        kind=ExpertReleaseUseRevocationKind.COMPATIBILITY,
        label="compatibility",
    )

    first = catalog.publish(
        expected_generation=projected,
        operation_id="publish_performance_release_use_revocation",
        objects=(performance,),
        dependency_closure_ids=(evidence_id,),
    ).generation
    committed = catalog.rebase(
        operation_id="publish_compatibility_release_use_revocation",
        objects=(compatibility,),
        dependency_closure_ids=(evidence_id,),
    ).generation
    view = catalog.read_generation(committed)

    assert view.facts.release_use_revocations == tuple(
        sorted((performance, compatibility), key=lambda event: event.revocation_id)
    )
    assert set(committed.active_entry_state_ids) == set(
        projected.active_entry_state_ids
    )
    assert all(
        event.revocation_id not in committed.active_entry_state_ids
        for event in (performance, compatibility)
    )
    assert all(
        not state.revocation_ids and not state.taint_source_ids
        for state in view.entry_states
    )


@pytest.mark.parametrize(
    ("scope_contract_id", "scope_id"),
    (
        (_external_id("expert-scope-contract", "another"), "ml_ai"),
        (None, "another_scope"),
    ),
)
def test_catalog_rejects_release_use_event_outside_exact_scope(
    tmp_path: Path,
    scope_contract_id: str | None,
    scope_id: str,
) -> None:
    catalog, projected, evidence_id = _projected_catalog(tmp_path)
    event = _revocation(
        evidence_id=evidence_id,
        scope_contract_id=scope_contract_id,
        scope_id=scope_id,
    )

    with pytest.raises(CatalogFactError, match="leaves the catalog scope"):
        catalog.publish(
            expected_generation=projected,
            operation_id="wrong_scope_release_use_revocation",
            objects=(event,),
            dependency_closure_ids=(evidence_id,),
        )

    assert catalog.store.read_current() == projected


def test_catalog_rejects_missing_release_use_evidence(tmp_path: Path) -> None:
    catalog, projected, _ = _projected_catalog(tmp_path)
    missing_evidence_id = _external_id("run-bundle", "missing")
    event = _revocation(evidence_id=missing_evidence_id)

    with pytest.raises(CatalogFactError, match="evidence closure is incomplete"):
        catalog.publish(
            expected_generation=projected,
            operation_id="missing_release_use_evidence",
            objects=(event,),
            dependency_closure_ids=(),
        )

    assert catalog.store.read_current() == projected


@pytest.mark.parametrize("finding_kind", ("revocation", "taint"))
def test_scientific_findings_cannot_target_release_use_events(
    tmp_path: Path,
    finding_kind: str,
) -> None:
    catalog, projected, evidence_id = _projected_catalog(tmp_path)
    event = _revocation(evidence_id=evidence_id)
    if finding_kind == "revocation":
        finding = CatalogRevocation.mint(
            subject_id=event.revocation_id,
            reason_code="invalid_cross_plane_target",
            rationale="Scientific revocation must not target release-use policy.",
            exact_evidence_refs=(evidence_id,),
        )
        message = "revocation subject is absent"
    else:
        finding = CatalogTaint.mint(
            subject_id=evidence_id,
            source_subject_id=event.revocation_id,
            reason_code="invalid_cross_plane_source",
            rationale="Scientific taint must not originate in release-use policy.",
            exact_evidence_refs=(evidence_id,),
        )
        message = "taint subject or source is absent"

    with pytest.raises(MissingReferenceError, match=message):
        catalog.publish(
            expected_generation=projected,
            operation_id=f"reject_release_use_{finding_kind}_crossing",
            objects=(event, finding),
            dependency_closure_ids=(evidence_id,),
        )

    assert catalog.store.read_current() == projected


def test_projection_cannot_use_release_use_event_as_episode_derivation(
    tmp_path: Path,
) -> None:
    fixture, projection = _project_real_bundle(tmp_path)
    scope = _scope_contract()
    catalog = CrossRunCatalog(
        tmp_path / "catalog",
        scope,
        fixture.settings.catalog,
    )
    event = _revocation(evidence_id=projection.source_bundle.bundle_id)
    original_episode = projection.episodes[0]
    episode_payload = original_episode.to_dict()
    episode_payload.pop("episode_id")
    episode_payload["derivation_refs"] = tuple(
        sorted((*original_episode.derivation_refs, event.revocation_id))
    )
    forged_episode = TransferEpisode.mint(**episode_payload)
    manifest_payload = projection.projection_manifest.to_dict()
    manifest_payload.pop("projection_manifest_id")
    manifest_payload["episode_ids"] = tuple(
        sorted(
            (
                forged_episode.episode_id
                if episode_id == original_episode.episode_id
                else episode_id
            )
            for episode_id in projection.projection_manifest.episode_ids
        )
    )
    manifest_payload["derivation_object_ids"] = tuple(
        sorted(
            (
                *projection.projection_manifest.derivation_object_ids,
                event.revocation_id,
            )
        )
    )
    forged_manifest = BundleProjectionManifest.mint(**manifest_payload)
    forged_facts = tuple(
        (
            forged_episode
            if record == original_episode
            else forged_manifest if record == projection.projection_manifest else record
        )
        for record in projection.catalog_facts
    )

    with pytest.raises(CatalogFactError, match="projection derivation event is absent"):
        catalog.publish(
            expected_generation=catalog.store.read_current(),
            operation_id="reject_release_use_derivation_crossing",
            objects=(*forged_facts, event),
            dependency_closure_ids=(),
        )

    assert catalog.store.read_current().generation_number == 0
