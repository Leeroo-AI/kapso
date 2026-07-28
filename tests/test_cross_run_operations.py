"""Operational commands remain thin and emit complete non-secret receipts."""

from __future__ import annotations

import json
from pathlib import Path

import kapso.cli as cli_module
import kapso.cross_run.operations as operations_module
from kapso.cli import main
from kapso.core.config import load_effective_config
from kapso.cross_run.catalog.service import CrossRunCatalog
from kapso.cross_run.contracts import CompletionState
from kapso.cross_run.operations import (
    GitHubOperationServices,
    capture_cross_run,
    publish_knowledge_cross_run,
)
from cross_run_capture_fixtures import make_capture_fixture
from test_cross_run_retrieval import source_fixture
from test_knowledge_snapshot_publisher import RecordingPublicationAuthority

_CONFIG_PATH = "src/kapso/config.yaml"
_COMMITTED_AT = "2026-07-27T12:00:00Z"


def test_capture_command_runs_the_real_pipeline_and_reports_exact_bundle(tmp_path):
    fixture_root = tmp_path / "workspace"
    fixture_root.mkdir()
    fixture = make_capture_fixture(fixture_root)
    request_path = tmp_path / "capture-request.json"
    request_path.write_text(
        json.dumps(_capture_request_payload(fixture.request), sort_keys=True),
        encoding="utf-8",
    )

    result = capture_cross_run(
        config_path=_CONFIG_PATH,
        mode="GENERIC",
        request_path=request_path,
    )

    assert result["operation"] == "capture"
    assert result["run_id"] == fixture.request.run_id
    assert result["campaign_id"] == fixture.request.campaign_id
    assert result["bundle_id"].startswith("run-bundle:sha256:")
    assert result["completion_state"] == CompletionState.STOPPED.value
    assert result["artifact_digests"]


def test_publish_knowledge_delegates_exact_empty_generation_to_m2(
    tmp_path,
    monkeypatch,
):
    settings = load_effective_config(_CONFIG_PATH, "GENERIC").cross_run
    scope_contract = source_fixture()[0]
    catalog_root = tmp_path / "catalog"
    catalog_root.mkdir()
    generation = CrossRunCatalog(
        catalog_root,
        scope_contract,
        settings.catalog,
    ).store.read_current()
    authority = RecordingPublicationAuthority()
    monkeypatch.setattr(
        operations_module,
        "_github_services",
        lambda _settings, _state_root: GitHubOperationServices(
            resolver=object(),
            materializer=object(),
            publisher=authority,
        ),
    )
    request_path = tmp_path / "publish.json"
    request_path.write_text(
        json.dumps(
            {
                "catalog_root": "catalog",
                "scope_contract": scope_contract.to_dict(),
                "expected_parent_sha": "a" * 40,
                "expected_current_snapshot_id": None,
                "committed_at": _COMMITTED_AT,
                "validation_closure_ids": [],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = publish_knowledge_cross_run(
        config_path=_CONFIG_PATH,
        mode="GENERIC",
        request_path=request_path,
        state_root=tmp_path,
    )

    assert result["operation"] == "publish-knowledge"
    assert result["snapshot_id"].startswith("knowledge-snapshot:sha256:")
    assert result["catalog_generation_id"] == generation.catalog_generation_id
    assert result["embedding"] is None
    assert result["commit_sha"] == "b" * 40


def test_cli_wrapper_only_parses_and_prints_operation_receipt(
    tmp_path,
    monkeypatch,
    capsysbinary,
):
    expected = {
        "operation": "inspect",
        "scope_id": "ml_ai",
        "artifacts": {},
    }
    calls = []

    def inspect(**arguments):
        calls.append(arguments)
        return expected

    monkeypatch.setattr(cli_module, "inspect_cross_run", inspect)

    main(
        [
            "cross-run",
            "inspect",
            "--config",
            _CONFIG_PATH,
            "--mode",
            "GENERIC",
            "--scope-id",
            "ml_ai",
            "--state-root",
            str(tmp_path),
        ]
    )

    assert calls == [
        {
            "config_path": _CONFIG_PATH,
            "mode": "GENERIC",
            "scope_id": "ml_ai",
            "state_root": Path(tmp_path),
        }
    ]
    assert json.loads(capsysbinary.readouterr().out) == expected


def _capture_request_payload(request):
    return {
        "workspace_dir": str(request.workspace_dir),
        "idea_archive_path": str(request.idea_archive_path),
        "scope_contract_id": request.scope_contract_id,
        "scope_id": request.scope_id,
        "run_id": request.run_id,
        "campaign_id": request.campaign_id,
        "configuration_fingerprint": request.configuration_fingerprint,
        "completion_state": request.completion_state.value,
        "started_at": request.started_at,
        "kapso_commit": request.kapso_commit,
        "launch_manifest_id": request.launch_manifest_id,
        "knowledge_snapshot_id": request.knowledge_snapshot_id,
        "expert_base_release_id": request.expert_base_release_id,
        "task_context_binding": request.task_context_binding.to_dict(),
        "artifact_environment": request.artifact_environment.to_dict(),
        "evaluation_fingerprints": [
            fingerprint.to_dict() for fingerprint in request.evaluation_fingerprints
        ],
        "run_log_paths": list(request.run_log_paths),
    }
