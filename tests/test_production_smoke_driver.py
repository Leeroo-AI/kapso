"""The production driver checkpoints one canonical cross-stage receipt."""

from __future__ import annotations

from pathlib import Path

import pytest

import kapso.cross_run.production_smoke as smoke_module
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
