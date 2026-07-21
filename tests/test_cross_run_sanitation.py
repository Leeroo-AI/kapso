import os

import pytest

from kapso.cross_run.canonical import canonical_json_bytes, parse_json_bytes

from kapso.cross_run.capture.exporter import RunCaptureExporter
from kapso.cross_run.capture.sanitation import SanitationGate, SanitationRejectedError
from kapso.cross_run.capture.validator import CaptureValidator
from kapso.cross_run.record_contracts import SanitationReport
from cross_run_capture_fixtures import make_capture_fixture


def sanitize_fixture(fixture, output_root=None):
    if output_root is None:
        output_root = (
            fixture.workspace / fixture.settings.capture.state_path / "sanitized"
        )
    exported = RunCaptureExporter(
        fixture.settings.capture,
        fixture.settings.sanitation,
    ).export(fixture.request)
    validated = CaptureValidator(
        fixture.settings.capture.score_comparison_tolerance
    ).validate(exported.path)
    return SanitationGate(
        fixture.settings.capture,
        fixture.settings.sanitation,
    ).sanitize(validated, output_root)


def test_sanitation_admits_safe_text_and_records_excluded_artifact_classes(tmp_path):
    fixture = make_capture_fixture(tmp_path, forbidden_artifacts=True)

    sanitized = sanitize_fixture(fixture)

    reasons = {item["reason"] for item in sanitized.report.excluded_paths}
    assert "artifact_class" in reasons
    assert "denied_path" in reasons
    assert all(not path.endswith((".pt", ".csv")) for path in sanitized.checksums)
    assert sanitized.report.status == "admitted"


def test_sanitation_rejects_secret_and_persists_rejection_report(tmp_path):
    fixture = make_capture_fixture(tmp_path, secret_source=True)

    with pytest.raises(SanitationRejectedError) as rejected:
        sanitize_fixture(fixture)

    assert rejected.value.report.status == "rejected"
    assert rejected.value.report_path.is_file()
    assert any(
        finding["code"] == "assigned_secret"
        for finding in rejected.value.report.findings
    )


def test_recorded_rejection_keeps_current_generation_and_marker_for_resume(tmp_path):
    fixture = make_capture_fixture(tmp_path, secret_source=True)
    exported = RunCaptureExporter(
        fixture.settings.capture,
        fixture.settings.sanitation,
    ).export(fixture.request)
    validated = CaptureValidator(
        fixture.settings.capture.score_comparison_tolerance
    ).validate(exported.path)
    current_marker = exported.path.parent / "current.json"

    with pytest.raises(SanitationRejectedError) as rejected:
        SanitationGate(
            fixture.settings.capture,
            fixture.settings.sanitation,
        ).sanitize(
            validated,
            fixture.workspace / fixture.settings.capture.state_path / "sanitized",
        )

    assert rejected.value.report_path.is_file()
    assert exported.path.is_dir()
    assert current_marker.is_file()


def test_sanitation_rejects_unapproved_source_license(tmp_path):
    fixture = make_capture_fixture(tmp_path, unapproved_license=True)

    with pytest.raises(SanitationRejectedError) as rejected:
        sanitize_fixture(fixture)

    assert any(
        finding["code"] == "unapproved_spdx_license"
        for finding in rejected.value.report.findings
    )


def test_sanitation_deduplicates_identical_license_findings_before_rejection(
    tmp_path,
):
    fixture = make_capture_fixture(tmp_path, multiple_unapproved_licenses=True)

    with pytest.raises(SanitationRejectedError) as rejected:
        sanitize_fixture(fixture)

    license_findings = tuple(
        finding
        for finding in rejected.value.report.findings
        if finding["code"] == "unapproved_spdx_license"
    )
    assert len(license_findings) == 1


def test_sanitation_excludes_configured_vcs_cache_data_and_evaluator_classes(
    tmp_path,
):
    fixture = make_capture_fixture(tmp_path, excluded_artifact_classes=True)

    sanitized = sanitize_fixture(fixture)

    excluded = {
        item["path"]
        for item in sanitized.report.excluded_paths
        if item["reason"] == "denied_path"
    }
    assert {
        ".svn/entries.json",
        ".cache/preprocessed.json",
        "__pycache__/metadata.json",
        "data/train.json",
        "training_data/samples.jsonl",
        "hidden_evaluator/test_cases.json",
        "evaluation/private.yaml",
        "weights/model.json",
    }.issubset(excluded)
    assert all(
        any(path.endswith(f"/files/{source_path}") for path in sanitized.checksums)
        for source_path in (
            "cache_adapter.py",
            "data.py",
            "dataset_reader.py",
            "evaluation_utils.py",
        )
    )


def test_sanitation_root_replacement_cannot_redirect_staging_creation(
    tmp_path,
    monkeypatch,
):
    fixture = make_capture_fixture(tmp_path)
    output_root = fixture.workspace / fixture.settings.capture.state_path / "sanitized"
    output_root.mkdir()
    detached_root = tmp_path / "detached-sanitized"
    victim = tmp_path / "victim"
    victim.mkdir()
    marker = victim / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    real_mkdir = os.mkdir
    swapped = False

    def replace_root_before_staging(path, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if str(path).startswith(".sanitized.") and not swapped:
            output_root.rename(detached_root)
            output_root.symlink_to(victim, target_is_directory=True)
            swapped = True
        return real_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "mkdir", replace_root_before_staging)

    with pytest.raises(ValueError, match="output directory was replaced"):
        sanitize_fixture(fixture, output_root)

    assert swapped
    assert marker.read_text(encoding="utf-8") == "keep"
    assert tuple(victim.iterdir()) == (marker,)


def test_sanitation_staging_replacement_cannot_redirect_writes_or_cleanup(
    tmp_path,
    monkeypatch,
):
    fixture = make_capture_fixture(tmp_path)
    output_root = fixture.workspace / fixture.settings.capture.state_path / "sanitized"
    output_root.mkdir()
    detached_staging = tmp_path / "detached-staging"
    victim = tmp_path / "victim"
    victim.mkdir()
    marker = victim / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    real_open = os.open
    swapped = False

    def replace_staging_before_write(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        staging_entries = tuple(output_root.glob(".sanitized.*"))
        if (
            flags & os.O_CREAT
            and path != ".sanitation.lock"
            and staging_entries
            and not swapped
        ):
            staging_entries[0].rename(detached_staging)
            staging_entries[0].symlink_to(victim, target_is_directory=True)
            swapped = True
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", replace_staging_before_write)

    with pytest.raises(ValueError, match="output directory was replaced"):
        sanitize_fixture(fixture, output_root)

    assert swapped
    assert marker.read_text(encoding="utf-8") == "keep"
    assert tuple(victim.iterdir()) == (marker,)


def test_sanitation_projects_raw_node_output_out_of_durable_content(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    sentinel = "RAW-NODE-CONTENT-MUST-NOT-SURVIVE"
    node = fixture.strategy.node_history[0]
    node.agent_output = sentinel
    node.evaluation_output = sentinel
    node.evaluation_script_path = sentinel
    node.code_diff = sentinel
    fixture.strategy.previous_errors = [sentinel]
    fixture.save_checkpoint("running")

    sanitized = sanitize_fixture(fixture)
    checkpoint_ref = sanitized.artifact_refs["checkpoint"]
    checkpoint = parse_json_bytes((sanitized.path / checkpoint_ref).read_bytes())
    checkpoint_bytes = (sanitized.path / checkpoint_ref).read_bytes()

    assert sentinel.encode() not in checkpoint_bytes
    assert checkpoint["strategy_state"]["previous_errors"] == []
    assert checkpoint["strategy_state"]["node_history"][0]["agent_output"] == ""


def test_sanitation_removes_free_form_observations_from_every_durable_ref(tmp_path):
    sentinel = "HIDDEN-EVALUATOR-EXAMPLE-MUST-NOT-SURVIVE"
    fixture = make_capture_fixture(
        tmp_path,
        raw_observation_sentinel=sentinel,
    )

    sanitized = sanitize_fixture(fixture)

    assert all(
        sentinel.encode("utf-8") not in (sanitized.path / path).read_bytes()
        for path in sanitized.checksums
    )
    history = parse_json_bytes(
        (sanitized.path / sanitized.artifact_refs["experiment_history"]).read_bytes()
    )
    record = history["records"][0]
    assert record["solution"] == fixture.strategy.node_history[0].solution
    assert record["metrics"] == {"quality": fixture.strategy.node_history[0].score}
    assert record["feedback"] == ""
    assert record["technical_difficulties"] == ""
    assert record["phase_telemetry"] == {}
    journal = parse_json_bytes(
        (sanitized.path / sanitized.artifact_refs["execution_event_journal"])
        .read_bytes()
        .strip()
    )
    expected_measurements = dict(record["metrics"])
    expected_measurements["raw_score"] = record["raw_score"]
    assert journal["measurements"] == expected_measurements
    checkpoint = parse_json_bytes(
        (sanitized.path / sanitized.artifact_refs["checkpoint"]).read_bytes()
    )
    assert checkpoint["current_feedback"] is None
    assert checkpoint["strategy_state"]["node_history"][0]["feedback"] == ""


def test_secret_only_in_raw_agent_output_is_removed_before_scanning(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    fixture.strategy.node_history[0].agent_output = (
        "API_KEY = 'super-secret-production-token'"
    )
    fixture.save_checkpoint("running")

    sanitized = sanitize_fixture(fixture)

    assert sanitized.report.status == "admitted"


def test_sanitation_projects_raw_ideation_agent_calls_from_both_archives(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    sentinel = "API_KEY = 'raw-ideation-secret-must-not-survive'"
    archive_path = fixture.strategy.idea_archive.path
    archive_source = parse_json_bytes(archive_path.read_bytes())
    for batch in archive_source["batches"]:
        for call in batch["generation_calls"]:
            call["output"] = sentinel
        if batch["selection_call"] is not None:
            batch["selection_call"]["output"] = sentinel
    archive_path.write_bytes(canonical_json_bytes(archive_source))
    fixture.save_checkpoint("running")

    sanitized = sanitize_fixture(fixture)
    archive_ref = sanitized.artifact_refs["idea_archive"]
    checkpoint_ref = sanitized.artifact_refs["checkpoint"]
    archive_payload = (sanitized.path / archive_ref).read_bytes()
    checkpoint_payload = (sanitized.path / checkpoint_ref).read_bytes()
    archive = parse_json_bytes(archive_payload)
    checkpoint = parse_json_bytes(checkpoint_payload)

    assert sanitized.report.status == "admitted"
    assert sentinel.encode() not in archive_payload
    assert sentinel.encode() not in checkpoint_payload
    for projected_archive in (
        archive,
        checkpoint["strategy_state"]["idea_archive_snapshot"],
    ):
        for batch in projected_archive["batches"]:
            calls = list(batch["generation_calls"])
            if batch["selection_call"] is not None:
                calls.append(batch["selection_call"])
            assert all(call["output"] == "" for call in calls)
            assert all(call["artifacts"] == [] for call in calls)
        assert all(
            idea["generation_artifacts"] == [] for idea in projected_archive["ideas"]
        )


def test_archive_projection_redacts_experiment_feedback_without_dropping_outcome(
    tmp_path,
):
    fixture = make_capture_fixture(tmp_path)
    sentinel = "HIDDEN-EVALUATOR-EXAMPLE-MUST-NOT-SURVIVE"
    archive = fixture.strategy.idea_archive.state.to_dict()
    archive["batches"][0]["evidence_snapshot"]["experiments"] = [
        {
            "feedback": sentinel,
            "technical_difficulty": sentinel,
            "proposal": "retain this proposal",
        }
    ]

    projected = SanitationGate._safe_archive_projection(archive)

    experiment = projected["batches"][0]["evidence_snapshot"]["experiments"][0]
    assert experiment == {
        "feedback": "",
        "technical_difficulty": None,
        "proposal": "retain this proposal",
    }
    assert projected["ideas"][0]["outcome"] == archive["ideas"][0]["outcome"]


def test_denied_credential_name_variants_are_excluded_from_git_payloads(tmp_path):
    fixture = make_capture_fixture(tmp_path, denied_name_variants=True)

    sanitized = sanitize_fixture(fixture)

    excluded = {
        item["path"]
        for item in sanitized.report.excluded_paths
        if item["reason"] == "denied_path"
    }
    assert {
        "credentials.json",
        ".gitconfig",
        ".git-credentials",
        "nested/prod_credentials.toml",
        ".netrc",
        ".npmrc",
        ".pypirc",
        ".aws/config.json",
        ".azure/settings.json",
        ".docker/config.json",
        ".gnupg/settings.json",
        ".ssh/config.json",
        "prod_credential.json",
        "secret_key.py",
    }.issubset(excluded)


@pytest.mark.parametrize(
    "finding",
    (
        {
            "code": "assigned_secret",
            "evidence_digest": "sha256:" + "a" * 64,
            "path": "../outside.txt",
            "severity": "reject",
        },
        {
            "code": "assigned_secret",
            "evidence_digest": "not-a-digest",
            "path": "source.py",
            "severity": "reject",
        },
        {
            "code": "assigned_secret",
            "evidence_digest": "sha256:" + "a" * 64,
            "path": "source.py",
            "severity": ["reject"],
        },
        {
            "code": "unknown_finding",
            "evidence_digest": "sha256:" + "a" * 64,
            "path": "source.py",
            "severity": "reject",
        },
    ),
)
def test_sanitation_report_rejects_malformed_finding_values(finding):
    with pytest.raises(ValueError, match="sanitation finding"):
        SanitationReport.mint(
            schema="kapso.sanitation_report.v1",
            capture_manifest_id="capture-manifest:sha256:" + "a" * 64,
            scope_id="ml_ai",
            task_family_id="language_model_post_training",
            policy_version="policy-v1",
            policy_fingerprint="sha256:" + "b" * 64,
            scanner_version="kapso.deterministic_text_scanner.v1",
            status="rejected",
            findings=(finding,),
            excluded_paths=(),
            taint_sources=(),
            admitted_refs={},
        )


@pytest.mark.parametrize(
    "exclusion",
    (
        {"path": "/absolute/source.py", "reason": "denied_path"},
        {"path": "source.py", "reason": ["denied_path"]},
        {"path": "source.py", "reason": "unknown_reason"},
    ),
)
def test_sanitation_report_rejects_malformed_exclusion_values(exclusion):
    with pytest.raises(ValueError, match="sanitation exclusion"):
        SanitationReport.mint(
            schema="kapso.sanitation_report.v1",
            capture_manifest_id="capture-manifest:sha256:" + "a" * 64,
            scope_id="ml_ai",
            task_family_id="language_model_post_training",
            policy_version="policy-v1",
            policy_fingerprint="sha256:" + "b" * 64,
            scanner_version="kapso.deterministic_text_scanner.v1",
            status="admitted",
            findings=(),
            excluded_paths=(exclusion,),
            taint_sources=(),
            admitted_refs={},
        )
