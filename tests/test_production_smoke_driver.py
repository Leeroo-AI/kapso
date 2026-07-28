"""The production driver checkpoints one canonical cross-stage receipt."""

from __future__ import annotations

from pathlib import Path

import pytest

import kapso.cross_run.production_smoke as smoke_module
from kapso.core.config import load_effective_config
from kapso.cross_run.canonical import parse_json_bytes
from kapso.cross_run.production_smoke import (
    ProductionSmokeError,
    run_production_smoke,
)

_CONFIG_PATH = "src/kapso/config.yaml"


def test_selected_stages_append_one_canonical_replayable_receipt(
    tmp_path,
    monkeypatch,
):
    calls = []

    def stage(**arguments):
        calls.append(arguments["stage"])
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


def test_preflight_evaluator_summary_exposes_missing_roots_without_keys():
    settings = load_effective_config(_CONFIG_PATH, "GENERIC").cross_run

    authority = smoke_module._expert_evaluator_authority(settings)

    assert authority["configured"] is False
    assert "expert_contract_evaluator" in authority["missing_issuer_ids"]
    assert authority["issuer_trust_roots"]["expert_contract_evaluator"] is None
    assert set(authority) == {
        "configured",
        "issuer_trust_roots",
        "missing_issuer_ids",
        "sealed_canary_trust_root",
    }
