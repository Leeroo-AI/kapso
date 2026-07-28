"""The production driver checkpoints one canonical cross-stage receipt."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import kapso.cross_run.production_smoke as smoke_module
from kapso.core.config import load_effective_config
from kapso.cross_run.canonical import content_id, parse_json_bytes
from kapso.cross_run.catalog.service import CrossRunCatalog
from kapso.cross_run.knowledge.publisher import KnowledgeSnapshotPublisher
from kapso.cross_run.production_smoke import (
    ProductionSmokeError,
    run_production_smoke,
)
from test_knowledge_snapshot_publisher import (
    DeterministicEmbeddingProvider,
    RecordingPublicationAuthority,
)

_CONFIG_PATH = "src/kapso/config.yaml"


def test_selected_stages_append_one_canonical_replayable_receipt(
    tmp_path,
    monkeypatch,
):
    calls = []
    observed_prior_evidence = []

    def stage(**arguments):
        calls.append(arguments["stage"])
        observed_prior_evidence.append(dict(arguments["prior_evidence"]))
        return {
            "stage_evidence_id": f"evidence-for-{arguments['stage']}",
        }

    monkeypatch.setattr(smoke_module, "_run_stage", stage)
    selected = ("preflight", "bootstrap-authorities", "github-read")

    first = run_production_smoke(
        config_path=_CONFIG_PATH,
        mode="GENERIC",
        state_root=tmp_path,
        stages=selected,
    )
    replayed = run_production_smoke(
        config_path=_CONFIG_PATH,
        mode="GENERIC",
        state_root=tmp_path,
        stages=selected,
    )

    assert calls == list(selected)
    assert observed_prior_evidence == [
        {},
        {"preflight": {"stage_evidence_id": "evidence-for-preflight"}},
        {
            "preflight": {"stage_evidence_id": "evidence-for-preflight"},
            "bootstrap-authorities": {
                "stage_evidence_id": "evidence-for-bootstrap-authorities"
            },
        },
    ]
    assert replayed == first
    assert tuple(item["stage"] for item in first["stage_receipts"]) == selected
    receipt_path = (
        tmp_path
        / ".kapso/cross_run/production_validation"
        / "production-smoke-receipt.json"
    )
    assert parse_json_bytes(receipt_path.read_bytes()) == first
    assert not (receipt_path.parent / ".production-smoke-receipt.next").exists()


def test_driver_rejects_out_of_order_stage_selection(tmp_path):
    with pytest.raises(ProductionSmokeError, match="out of order"):
        run_production_smoke(
            config_path=_CONFIG_PATH,
            mode="GENERIC",
            state_root=tmp_path,
            stages=("github-read", "preflight"),
        )


def test_driver_fails_loud_on_corrupt_durable_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(
        smoke_module,
        "_run_stage",
        lambda **_arguments: {"passed": True},
    )
    run_production_smoke(
        config_path=_CONFIG_PATH,
        mode="GENERIC",
        state_root=tmp_path,
        stages=("preflight",),
    )
    receipt_path = Path(tmp_path) / (
        ".kapso/cross_run/production_validation/production-smoke-receipt.json"
    )
    receipt_path.write_bytes(b"{not-json}\n")

    with pytest.raises(ValueError):
        run_production_smoke(
            config_path=_CONFIG_PATH,
            mode="GENERIC",
            state_root=tmp_path,
            stages=("preflight",),
        )


def test_synthetic_projection_is_one_admitted_domain_neutral_bundle():
    settings = load_effective_config(_CONFIG_PATH, "GENERIC").cross_run
    fixture, _fixture_digest = smoke_module._load_fixture(settings)
    scope_contract = smoke_module.ExpertScopeContract.from_dict(
        fixture["scope_contract"]
    )
    projection = smoke_module._synthetic_projection(
        settings,
        fixture,
        scope_contract,
    )

    assert projection.sanitation_report.status == "admitted"
    assert projection.source_bundle.scope_id == "ml_ai"
    assert projection.episodes == ()
    assert len(projection.prior_ideas) == 1
    assert projection.catalog_facts[-1] == projection.projection_manifest


def test_production_ideation_output_must_cite_the_retrieved_prior():
    expected = "prior-idea:sha256:" + "1" * 64
    output = (
        '{"idea":"change one variable","mechanism":"preserve causality",'
        f'"prior_record_id":"{expected}"}}'
    )

    assert (
        smoke_module._validate_production_ideation_output(
            output,
            expected,
        )["prior_record_id"]
        == expected
    )
    with pytest.raises(ProductionSmokeError, match="selected prior idea"):
        smoke_module._validate_production_ideation_output(
            output,
            "prior-idea:sha256:" + "2" * 64,
        )


def test_expert_bootstrap_exposes_every_scope_task_binding():
    settings = load_effective_config(_CONFIG_PATH, "GENERIC").cross_run
    fixture, _fixture_digest = smoke_module._load_fixture(settings)
    scope_contract = smoke_module.ExpertScopeContract.from_dict(
        fixture["scope_contract"]
    )
    bindings = smoke_module._scope_task_bindings(scope_contract)

    assert tuple(
        (binding.task_family_id, binding.task_adapter_id) for binding in bindings
    ) == (
        ("language_model_post_training", "posttrain"),
        ("relational_tabular_prediction", "relbench"),
    )


def test_task_adapter_bootstrap_precedes_expert_proposal():
    stages = smoke_module.production_smoke_stage_names()

    assert stages.index("task-adapter-bootstrap") < stages.index("expert-proposal")
    assert stages.index("expert-proposal") < stages.index(
        "expert-validation-enrollment"
    )


def test_expert_validation_enrollment_uses_exact_proposal_candidate(
    tmp_path,
    monkeypatch,
):
    smoke_root = tmp_path / "smoke"
    smoke_root.mkdir()
    candidate_id = content_id("expert-candidate", {"candidate": "production"})
    observed = {}

    def validate(**arguments):
        observed.update(arguments)
        request = parse_json_bytes(arguments["request_path"].read_bytes())
        assert request == {
            "candidate_id": candidate_id,
            "evaluator_result": None,
            "expected_transition_id": None,
        }
        return {
            "operation": "validate-expert",
            "candidate_id": candidate_id,
            "validation_attempt_id": "attempt-id",
            "transition_id": "transition-id",
            "validation_state_id": "state-id",
            "next_stage": "contract_schema",
        }

    monkeypatch.setattr(smoke_module, "validate_expert_cross_run", validate)
    result = smoke_module._expert_validation_enrollment_smoke(
        _CONFIG_PATH,
        "GENERIC",
        smoke_root,
        {
            "expert-proposal": {
                "candidate_id": candidate_id,
                "proposal_skipped": False,
            }
        },
    )

    assert observed["config_path"] == _CONFIG_PATH
    assert observed["mode"] == "GENERIC"
    assert observed["state_root"] == smoke_root
    assert result == {
        "candidate_id": candidate_id,
        "validation_attempt_id": "attempt-id",
        "transition_id": "transition-id",
        "validation_state_id": "state-id",
        "next_stage": "contract_schema",
        "validation_skipped": False,
    }


def test_expert_validation_enrollment_skips_authenticated_current_release(tmp_path):
    release_id = content_id("expert-base-release", {"release": "current"})

    assert smoke_module._expert_validation_enrollment_smoke(
        _CONFIG_PATH,
        "GENERIC",
        tmp_path,
        {
            "expert-proposal": {
                "existing_release_id": release_id,
                "proposal_skipped": True,
            }
        },
    ) == {
        "existing_release_id": release_id,
        "validation_skipped": True,
    }


def test_expert_validation_enrollment_requires_proposal_evidence(tmp_path):
    with pytest.raises(ProductionSmokeError, match="proposal evidence"):
        smoke_module._expert_validation_enrollment_smoke(
            _CONFIG_PATH,
            "GENERIC",
            tmp_path,
            {},
        )


def test_preflight_evaluator_summary_exposes_missing_roots_without_keys():
    settings = load_effective_config(_CONFIG_PATH, "GENERIC").cross_run

    authority = smoke_module._expert_evaluator_authority(settings)

    assert authority["configured"] is False
    assert "expert_contract_evaluator" in authority["missing_issuer_ids"]
    assert authority["issuer_trust_roots"]["expert_contract_evaluator"] is None
    assert "expert_source_replay_evaluator" not in authority["issuer_trust_roots"]
    assert "expert_release_matrix_evaluator" not in authority["issuer_trust_roots"]
    assert set(authority) == {
        "configured",
        "issuer_trust_roots",
        "missing_issuer_ids",
        "sealed_canary_trust_root",
    }


def test_clean_root_imports_current_snapshot_and_mints_one_direct_successor(
    tmp_path,
):
    settings = load_effective_config(_CONFIG_PATH, "GENERIC").cross_run
    old_settings = replace(
        settings,
        expert=replace(
            settings.expert,
            architect_id="production_transport_architect_old",
        ),
    )
    fixture, _fixture_digest = smoke_module._load_fixture(settings)
    scope_contract = smoke_module.ExpertScopeContract.from_dict(
        fixture["scope_contract"]
    )
    old_projection = smoke_module._synthetic_projection(
        old_settings,
        fixture,
        scope_contract,
    )
    old_catalog = CrossRunCatalog(
        tmp_path / "old-catalog",
        scope_contract,
        settings.catalog,
    )
    old_generation = old_catalog.publish_projection(
        old_catalog.store.read_current(),
        old_projection,
    ).generation
    publisher = KnowledgeSnapshotPublisher(
        RecordingPublicationAuthority(),
        settings.github,
        settings.knowledge,
        DeterministicEmbeddingProvider(settings.knowledge.embeddings),
    )
    old_package = publisher.build(
        scope_contract,
        old_generation,
        old_catalog.store.read_object_bytes,
        parent_snapshot_ids=(content_id("knowledge-snapshot", {"transport": "empty"}),),
        sanitation_policy_version=settings.sanitation.policy_version,
        retrieval_policy_version="kapso.retrieval.v1",
        published_at="2026-07-28T08:00:00Z",
        publisher_attestation={"issuer": "test-publisher"},
    ).package

    successor = smoke_module._synthetic_projection_for_snapshot(
        settings,
        fixture,
        scope_contract,
        old_package,
    )

    assert successor.source_bundle.capture_generation == 1
    assert successor.source_bundle.supersedes_bundle_id == (
        old_projection.source_bundle.bundle_id
    )
    assert successor.prior_ideas[0].supersedes_projection_id == (
        old_projection.prior_ideas[0].prior_idea_id
    )
    new_catalog = CrossRunCatalog(
        tmp_path / "new-catalog",
        scope_contract,
        settings.catalog,
    )
    seeded = smoke_module._seed_catalog_from_snapshot(new_catalog, old_package)
    successor_generation = new_catalog.publish_projection(
        seeded,
        successor,
    ).generation
    successor_package = publisher.build(
        scope_contract,
        successor_generation,
        new_catalog.store.read_object_bytes,
        parent_snapshot_ids=(old_package.manifest.snapshot_id,),
        sanitation_policy_version=settings.sanitation.policy_version,
        retrieval_policy_version="kapso.retrieval.v1",
        published_at="2026-07-28T08:00:00Z",
        publisher_attestation={"issuer": "test-publisher"},
    ).package

    recovered = smoke_module._synthetic_projection_for_snapshot(
        settings,
        fixture,
        scope_contract,
        successor_package,
    )

    assert recovered == successor
