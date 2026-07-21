import os
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from kapso.cross_run.capture.bundle import (
    RunBundlePublicationError,
    RunBundlePublisher,
)
from kapso.cross_run.capture.exporter import RunCaptureExporter
from kapso.cross_run.capture.sanitation import (
    SanitationGate,
    SanitationRejectedError,
)
from kapso.cross_run.capture.safety import read_restricted_regular_file
from kapso.cross_run.capture.validator import CaptureValidator
from kapso.cross_run.canonical import canonical_json_bytes, tree_or_blob_digest
from kapso.cross_run.contracts import CompletionState, RunBundle
from kapso.cross_run.record_contracts import SanitationReport
from cross_run_capture_fixtures import make_capture_fixture


def pipeline(fixture, publisher):
    exported = RunCaptureExporter(
        fixture.settings.capture,
        fixture.settings.sanitation,
    ).export(fixture.request)
    validated = CaptureValidator(
        fixture.settings.capture.score_comparison_tolerance
    ).validate(exported.path)
    sanitized = SanitationGate(
        fixture.settings.capture, fixture.settings.sanitation
    ).sanitize(
        validated,
        fixture.workspace / fixture.settings.capture.state_path / "sanitized",
    )
    return validated, publisher.publish(validated, sanitized)


def test_bundle_store_round_trip_is_content_addressed_and_idempotent(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    publisher = RunBundlePublisher(
        fixture.workspace / fixture.settings.capture.state_path / "bundles",
        fixture.settings.capture,
        fixture.settings.sanitation,
    )
    validated, first = pipeline(fixture, publisher)
    sanitized_again = SanitationGate(
        fixture.settings.capture, fixture.settings.sanitation
    ).sanitize(
        validated,
        fixture.workspace / fixture.settings.capture.state_path / "sanitized",
    )
    second = publisher.publish(validated, sanitized_again)

    assert second.manifest.bundle_id == first.manifest.bundle_id
    assert first.read_ref(first.manifest.experiment_history_ref)
    assert first.read_ref("sanitation_report.json")


def test_later_capture_supersedes_bundle_and_prunes_old_quarantine(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    publisher = RunBundlePublisher(
        fixture.workspace / fixture.settings.capture.state_path / "bundles",
        fixture.settings.capture,
        fixture.settings.sanitation,
    )
    _, first = pipeline(fixture, publisher)
    fixture.save_checkpoint("completed")
    fixture.request = replace(
        fixture.request,
        completion_state=CompletionState.COMPLETE,
    )

    _, second = pipeline(fixture, publisher)

    assert second.manifest.supersedes_bundle_id == first.manifest.bundle_id
    assert second.manifest.capture_generation == 1
    generations = list(
        (fixture.workspace / fixture.settings.capture.quarantine_path).glob(
            "runs/*/generation-*"
        )
    )
    assert len(generations) == fixture.settings.capture.quarantine_retention_generations


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("run_id", "run_" + "b" * 32),
        ("started_at", "2024-01-01T00:00:00Z"),
    ),
)
def test_bundle_current_cannot_cross_run_lineage(tmp_path, field, value):
    fixture = make_capture_fixture(tmp_path)
    publisher = RunBundlePublisher(
        fixture.workspace / fixture.settings.capture.state_path / "bundles",
        fixture.settings.capture,
        fixture.settings.sanitation,
    )
    validated, first = pipeline(fixture, publisher)
    manifest_values = first.manifest.to_dict()
    manifest_values.pop("bundle_id")
    manifest_values[field] = value
    wrong_bundle = RunBundle.mint(**manifest_values)
    publisher._commit_bundle(wrong_bundle, first.object_refs)
    marker_path = next((publisher.root / "runs").glob("*/current.json"))
    marker_path.write_bytes(
        canonical_json_bytes(
            {
                "bundle_id": wrong_bundle.bundle_id,
                "capture_generation": wrong_bundle.capture_generation,
            }
        )
        + b"\n"
    )
    marker_path.chmod(0o600)

    with pytest.raises(RunBundlePublicationError, match="another run identity"):
        publisher._load_current(validated)


def test_bundle_contract_binds_kapso_commit_to_artifact_environment(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    publisher = RunBundlePublisher(
        fixture.workspace / fixture.settings.capture.state_path / "bundles",
        fixture.settings.capture,
        fixture.settings.sanitation,
    )
    _, bundle = pipeline(fixture, publisher)
    manifest_values = bundle.manifest.to_dict()
    manifest_values.pop("bundle_id")
    manifest_values["kapso_commit"] = "f" * 40

    with pytest.raises(ValueError, match="another Kapso commit"):
        RunBundle.mint(**manifest_values)


def test_rejected_generation_preserves_sequence_for_corrected_supersession(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    publisher = RunBundlePublisher(
        fixture.workspace / fixture.settings.capture.state_path / "bundles",
        fixture.settings.capture,
        fixture.settings.sanitation,
    )
    _, first = pipeline(fixture, publisher)
    run_log = fixture.workspace / "run.txt"
    run_log.write_text(
        "API_KEY = 'super-secret-production-token'\n",
        encoding="utf-8",
    )
    request_with_log = replace(fixture.request, run_log_paths=("run.txt",))
    rejected_export = RunCaptureExporter(
        fixture.settings.capture,
        fixture.settings.sanitation,
    ).export(request_with_log)
    rejected_capture = CaptureValidator(
        fixture.settings.capture.score_comparison_tolerance
    ).validate(rejected_export.path)

    with pytest.raises(SanitationRejectedError):
        SanitationGate(
            fixture.settings.capture,
            fixture.settings.sanitation,
        ).sanitize(
            rejected_capture,
            fixture.workspace / fixture.settings.capture.state_path / "sanitized",
        )

    assert rejected_export.manifest.capture_generation == 1
    assert rejected_export.path.is_dir()
    run_log.write_text("safe observation\n", encoding="utf-8")
    corrected_export = RunCaptureExporter(
        fixture.settings.capture,
        fixture.settings.sanitation,
    ).export(request_with_log)
    corrected_capture = CaptureValidator(
        fixture.settings.capture.score_comparison_tolerance
    ).validate(corrected_export.path)
    corrected_sanitized = SanitationGate(
        fixture.settings.capture,
        fixture.settings.sanitation,
    ).sanitize(
        corrected_capture,
        fixture.workspace / fixture.settings.capture.state_path / "sanitized",
    )
    corrected = publisher.publish(corrected_capture, corrected_sanitized)

    assert corrected.manifest.capture_generation == 2
    assert corrected.manifest.supersedes_bundle_id == first.manifest.bundle_id


def test_publisher_rejects_incomplete_or_extra_sanitized_closure(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    exported = RunCaptureExporter(
        fixture.settings.capture,
        fixture.settings.sanitation,
    ).export(fixture.request)
    validated = CaptureValidator(
        fixture.settings.capture.score_comparison_tolerance
    ).validate(exported.path)
    sanitized = SanitationGate(
        fixture.settings.capture, fixture.settings.sanitation
    ).sanitize(
        validated,
        fixture.workspace / fixture.settings.capture.state_path / "sanitized",
    )
    publisher = RunBundlePublisher(
        fixture.workspace / fixture.settings.capture.state_path / "bundles",
        fixture.settings.capture,
        fixture.settings.sanitation,
    )
    nested_ref = next(
        ref
        for name, ref in sanitized.artifact_refs.items()
        if name.startswith("source:")
    )
    incomplete_checksums = dict(sanitized.checksums)
    incomplete_checksums.pop(nested_ref)

    with pytest.raises(RunBundlePublicationError, match="closure"):
        publisher.publish(
            validated,
            replace(sanitized, checksums=incomplete_checksums),
        )

    rogue = sanitized.path / "rogue.txt"
    rogue.write_text("rogue", encoding="utf-8")
    rogue.chmod(0o600)
    with pytest.raises(RunBundlePublicationError, match="file closure"):
        publisher.publish(validated, sanitized)


def test_publisher_binds_in_memory_report_to_persisted_report_bytes(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    exported = RunCaptureExporter(
        fixture.settings.capture,
        fixture.settings.sanitation,
    ).export(fixture.request)
    validated = CaptureValidator(
        fixture.settings.capture.score_comparison_tolerance
    ).validate(exported.path)
    sanitized = SanitationGate(
        fixture.settings.capture, fixture.settings.sanitation
    ).sanitize(
        validated,
        fixture.workspace / fixture.settings.capture.state_path / "sanitized",
    )
    publisher = RunBundlePublisher(
        fixture.workspace / fixture.settings.capture.state_path / "bundles",
        fixture.settings.capture,
        fixture.settings.sanitation,
    )
    values = sanitized.report.to_dict()
    values.pop("report_id")
    values["taint_sources"] = ("unclassified_license",)
    substituted = SanitationReport.mint(**values)
    report_path = sanitized.path / "sanitation_report.json"
    report_payload = substituted.to_json_bytes()
    report_path.write_bytes(report_payload)
    report_path.chmod(0o600)
    checksums = dict(sanitized.checksums)
    checksums["sanitation_report.json"] = tree_or_blob_digest(report_payload)

    with pytest.raises(RunBundlePublicationError, match="report bytes"):
        publisher.publish(
            validated,
            replace(sanitized, checksums=checksums),
        )


def test_publisher_binds_report_to_every_effective_sanitation_setting(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    exported = RunCaptureExporter(
        fixture.settings.capture,
        fixture.settings.sanitation,
    ).export(fixture.request)
    validated = CaptureValidator(
        fixture.settings.capture.score_comparison_tolerance
    ).validate(exported.path)
    sanitized = SanitationGate(
        fixture.settings.capture, fixture.settings.sanitation
    ).sanitize(
        validated,
        fixture.workspace / fixture.settings.capture.state_path / "sanitized",
    )
    changed_settings = replace(
        fixture.settings.sanitation,
        allowed_spdx_licenses=tuple(
            sorted((*fixture.settings.sanitation.allowed_spdx_licenses, "ISC"))
        ),
    )
    publisher = RunBundlePublisher(
        fixture.workspace / fixture.settings.capture.state_path / "bundles",
        fixture.settings.capture,
        changed_settings,
    )

    with pytest.raises(RunBundlePublicationError, match="policy settings"):
        publisher.publish(validated, sanitized)


def test_publisher_rejects_replaced_sanitized_staging_without_deleting_victim(
    tmp_path,
):
    fixture = make_capture_fixture(tmp_path)
    exported = RunCaptureExporter(
        fixture.settings.capture,
        fixture.settings.sanitation,
    ).export(fixture.request)
    validated = CaptureValidator(
        fixture.settings.capture.score_comparison_tolerance
    ).validate(exported.path)
    sanitized = SanitationGate(
        fixture.settings.capture, fixture.settings.sanitation
    ).sanitize(
        validated,
        fixture.workspace / fixture.settings.capture.state_path / "sanitized",
    )
    publisher = RunBundlePublisher(
        fixture.workspace / fixture.settings.capture.state_path / "bundles",
        fixture.settings.capture,
        fixture.settings.sanitation,
    )
    detached = tmp_path / "detached-sanitized"
    victim = tmp_path / "victim"
    victim.mkdir()
    marker = victim / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    sanitized.path.rename(detached)
    victim.rename(sanitized.path)

    with pytest.raises(RunBundlePublicationError, match="staging identity"):
        publisher.publish(validated, sanitized)

    assert (sanitized.path / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_restricted_read_pins_parent_before_symlink_replacement(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "root"
    nested = root / "nested"
    nested.mkdir(parents=True)
    trusted = nested / "artifact.txt"
    trusted.write_bytes(b"trusted")
    trusted.chmod(0o600)
    attacker = tmp_path / "attacker"
    attacker.mkdir()
    replacement = attacker / "artifact.txt"
    replacement.write_bytes(b"attacker")
    replacement.chmod(0o600)
    detached = root / "detached"
    real_open = os.open
    swapped = False

    def replace_parent_then_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if Path(path).name == "artifact.txt" and not swapped:
            nested.rename(detached)
            nested.symlink_to(attacker, target_is_directory=True)
            swapped = True
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", replace_parent_then_open)

    payload = read_restricted_regular_file(
        root,
        "nested/artifact.txt",
        RunBundlePublicationError,
    )

    assert swapped
    assert payload == b"trusted"


def test_restricted_read_rejects_root_replaced_by_symlink_before_open(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "root"
    root.mkdir()
    artifact = root / "artifact.txt"
    artifact.write_bytes(b"trusted")
    artifact.chmod(0o600)
    attacker = tmp_path / "attacker"
    attacker.mkdir()
    outside_secret = attacker / "artifact.txt"
    outside_secret.write_bytes(b"outside-secret")
    outside_secret.chmod(0o600)
    detached = tmp_path / "detached-root"
    real_open = os.open
    swapped = False

    def replace_root_then_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if not swapped:
            root.rename(detached)
            root.symlink_to(attacker, target_is_directory=True)
            swapped = True
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", replace_root_then_open)

    with pytest.raises(OSError):
        read_restricted_regular_file(
            root,
            "artifact.txt",
            RunBundlePublicationError,
        )

    assert swapped
    assert outside_secret.read_bytes() == b"outside-secret"


def test_publish_rejects_forged_cleanup_roots_without_deleting_them(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    exported = RunCaptureExporter(
        fixture.settings.capture,
        fixture.settings.sanitation,
    ).export(fixture.request)
    validated = CaptureValidator(
        fixture.settings.capture.score_comparison_tolerance
    ).validate(exported.path)
    sanitized = SanitationGate(
        fixture.settings.capture, fixture.settings.sanitation
    ).sanitize(
        validated,
        fixture.workspace / fixture.settings.capture.state_path / "sanitized",
    )
    publisher = RunBundlePublisher(
        fixture.workspace / fixture.settings.capture.state_path / "bundles",
        fixture.settings.capture,
        fixture.settings.sanitation,
    )
    victim_sanitized = tmp_path / "victim" / ".sanitized.forged"
    shutil.copytree(sanitized.path, victim_sanitized)

    with pytest.raises(RunBundlePublicationError, match="sanitized path"):
        publisher.publish(
            validated,
            replace(sanitized, path=victim_sanitized),
        )

    victim_run = tmp_path / "victim-run"
    victim_generations = tuple(
        victim_run / f"generation-{generation:020d}" for generation in range(5)
    )
    for generation in victim_generations:
        generation.mkdir(parents=True)
        (generation / "keep.txt").write_text("keep", encoding="utf-8")
    forged_capture = replace(validated, path=victim_generations[-1])

    with pytest.raises(RunBundlePublicationError, match="capture path"):
        publisher.publish(forged_capture, sanitized)

    assert victim_sanitized.is_dir()
    assert all((generation / "keep.txt").is_file() for generation in victim_generations)


def test_publisher_rejects_store_outside_configured_state_path(tmp_path):
    fixture = make_capture_fixture(tmp_path)

    with pytest.raises(RunBundlePublicationError, match="configured workspace"):
        RunBundlePublisher(
            tmp_path / "unconfigured" / "bundles",
            fixture.settings.capture,
            fixture.settings.sanitation,
        )
