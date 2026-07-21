import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

import kapso.cross_run.capture.bundle as bundle_module
from kapso.cross_run.capture.bundle import (
    RunBundlePublicationError,
    RunBundlePublisher,
    RunBundleStore,
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


def sanitize_capture(fixture):
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
    return validated, sanitized


def pipeline(fixture, publisher):
    validated, sanitized = sanitize_capture(fixture)
    return validated, publisher.publish(validated, sanitized)


def test_bundle_store_round_trip_is_content_addressed_and_idempotent(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    publisher = RunBundlePublisher(
        fixture.workspace / fixture.settings.capture.state_path,
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


def test_bundle_store_is_read_only_exact_and_snapshots_verified_bytes(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    state_root = fixture.workspace / fixture.settings.capture.state_path
    publisher = RunBundlePublisher(
        state_root,
        fixture.settings.capture,
        fixture.settings.sanitation,
    )
    _, published = pipeline(fixture, publisher)
    store = RunBundleStore(
        state_root,
        fixture.settings.capture,
        fixture.settings.sanitation,
    )

    observed = store.require_exact(published.manifest.bundle_id)
    bundle_path = (
        state_root / "bundles" / published.manifest.bundle_id.rsplit(":", 1)[1]
    )
    assert {path.name for path in bundle_path.iterdir()} == {"manifest.json"}
    assert not (state_root / "bundles" / "bundles").exists()
    assert store.read_exact("run-bundle:sha256:" + "0" * 64) is None
    shutil.rmtree(state_root / "runs")
    assert (
        store.require_exact(published.manifest.bundle_id).manifest == observed.manifest
    )

    relative_path = published.manifest.idea_archive_ref
    digest = published.manifest.checksums[relative_path]
    object_path = state_root / "objects" / "sha256" / digest[7:]
    original = observed.read_ref(relative_path)
    object_path.chmod(0o600)
    object_path.write_bytes(b"changed after verified read")
    object_path.chmod(0o400)

    assert observed.read_ref(relative_path) == original
    with pytest.raises(RunBundlePublicationError, match="digest changed"):
        store.require_exact(published.manifest.bundle_id)


def test_bundle_store_rejects_noncanonical_or_mutable_control_closure(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    state_root = fixture.workspace / fixture.settings.capture.state_path
    publisher = RunBundlePublisher(
        state_root,
        fixture.settings.capture,
        fixture.settings.sanitation,
    )
    _, published = pipeline(fixture, publisher)
    store = RunBundleStore(
        state_root,
        fixture.settings.capture,
        fixture.settings.sanitation,
    )
    bundle_path = (
        state_root / "bundles" / published.manifest.bundle_id.rsplit(":", 1)[1]
    )
    manifest_path = bundle_path / "manifest.json"

    bundle_path.chmod(0o700)
    manifest_path.chmod(0o600)
    manifest_path.write_bytes(published.manifest.to_json_bytes() + b"\n")
    manifest_path.chmod(0o400)
    bundle_path.chmod(0o500)
    with pytest.raises(RunBundlePublicationError, match="not canonical"):
        store.require_exact(published.manifest.bundle_id)

    bundle_path.chmod(0o700)
    manifest_path.chmod(0o600)
    manifest_path.write_bytes(published.manifest.to_json_bytes())
    manifest_path.chmod(0o400)
    (bundle_path / "extra.json").write_bytes(b"{}")
    (bundle_path / "extra.json").chmod(0o400)
    bundle_path.chmod(0o500)
    with pytest.raises(RunBundlePublicationError, match="control closure"):
        store.require_exact(published.manifest.bundle_id)


def test_bundle_store_enforces_configured_entry_byte_and_immutability_bounds(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    state_root = fixture.workspace / fixture.settings.capture.state_path
    publisher = RunBundlePublisher(
        state_root,
        fixture.settings.capture,
        fixture.settings.sanitation,
    )
    _, published = pipeline(fixture, publisher)

    root_entry_count = len(tuple(state_root.iterdir()))
    assert len(published.manifest.checksums) > root_entry_count
    entry_limited = RunBundleStore(
        state_root,
        replace(
            fixture.settings.capture,
            bundle_entry_limit=root_entry_count,
        ),
        fixture.settings.sanitation,
    )
    with pytest.raises(RunBundlePublicationError, match="entry limit"):
        entry_limited.require_exact(published.manifest.bundle_id)

    byte_limited = RunBundleStore(
        state_root,
        replace(fixture.settings.capture, bundle_asset_size_bytes=1),
        fixture.settings.sanitation,
    )
    with pytest.raises(RunBundlePublicationError, match="byte limit"):
        byte_limited.require_exact(published.manifest.bundle_id)

    relative_path = published.manifest.idea_archive_ref
    digest = published.manifest.checksums[relative_path]
    object_path = state_root / "objects" / "sha256" / digest[7:]
    object_path.chmod(0o600)
    with pytest.raises(RunBundlePublicationError, match="file identity"):
        publisher.store.require_exact(published.manifest.bundle_id)


@pytest.mark.parametrize("directory_name", ("objects", "bundles", "runs"))
def test_publisher_rejects_preexisting_layout_symlink_without_writing_victim(
    tmp_path,
    directory_name,
):
    fixture = make_capture_fixture(tmp_path)
    state_root = fixture.workspace / fixture.settings.capture.state_path
    state_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    state_root.chmod(0o700)
    victim = tmp_path / f"{directory_name}-victim"
    victim.mkdir()
    (state_root / directory_name).symlink_to(victim, target_is_directory=True)

    with pytest.raises(OSError):
        RunBundlePublisher(
            state_root,
            fixture.settings.capture,
            fixture.settings.sanitation,
        )

    assert tuple(victim.iterdir()) == ()


@pytest.mark.parametrize("directory_name", ("objects", "bundles", "runs"))
def test_publisher_rejects_replaced_authoritative_store_directory(
    tmp_path,
    directory_name,
):
    fixture = make_capture_fixture(tmp_path)
    state_root = fixture.workspace / fixture.settings.capture.state_path
    publisher = RunBundlePublisher(
        state_root,
        fixture.settings.capture,
        fixture.settings.sanitation,
    )
    pipeline(fixture, publisher)
    authoritative = state_root / directory_name
    authoritative.rename(state_root / f"{directory_name}-detached")
    authoritative.mkdir(mode=0o700)

    with pytest.raises(RunBundlePublicationError, match="directory identity changed"):
        pipeline(fixture, publisher)

    assert tuple(authoritative.iterdir()) == ()


def test_publisher_does_not_treat_dangling_current_marker_as_absent(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    state_root = fixture.workspace / fixture.settings.capture.state_path
    publisher = RunBundlePublisher(
        state_root,
        fixture.settings.capture,
        fixture.settings.sanitation,
    )
    pipeline(fixture, publisher)
    marker_path = next((state_root / "runs").glob("*/current.json"))
    marker_path.unlink()
    marker_path.symlink_to(tmp_path / "absent-marker-target")

    with pytest.raises(OSError):
        pipeline(fixture, publisher)

    assert marker_path.is_symlink()


def test_bundle_store_rejects_hard_linked_content_object(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    state_root = fixture.workspace / fixture.settings.capture.state_path
    publisher = RunBundlePublisher(
        state_root,
        fixture.settings.capture,
        fixture.settings.sanitation,
    )
    _, published = pipeline(fixture, publisher)
    digest = published.manifest.checksums[published.manifest.idea_archive_ref]
    object_path = state_root / "objects" / "sha256" / digest[7:]
    detached_path = tmp_path / "detached-object"
    object_path.rename(detached_path)
    os.link(detached_path, object_path)

    with pytest.raises(RunBundlePublicationError, match="file identity"):
        publisher.store.require_exact(published.manifest.bundle_id)

    assert detached_path.stat().st_nlink == 2


def test_oversized_capture_cannot_publish_partial_authority(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    validated, sanitized = sanitize_capture(fixture)
    state_root = fixture.workspace / fixture.settings.capture.state_path
    publisher = RunBundlePublisher(
        state_root,
        replace(fixture.settings.capture, bundle_asset_size_bytes=1),
        fixture.settings.sanitation,
    )

    with pytest.raises(RunBundlePublicationError, match="byte limit"):
        publisher.publish(validated, sanitized)

    assert tuple((state_root / "objects" / "sha256").iterdir()) == ()
    assert tuple((state_root / "bundles").iterdir()) == ()
    assert tuple((state_root / "runs").iterdir()) == ()


def test_concurrent_first_publishers_serialize_to_one_bundle(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    state_root = fixture.workspace / fixture.settings.capture.state_path

    def construct_publisher(worker_name):
        assert worker_name in {"first", "second"}
        return RunBundlePublisher(
            state_root,
            fixture.settings.capture,
            fixture.settings.sanitation,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        publishers = tuple(executor.map(construct_publisher, ("first", "second")))
    validated, first_sanitized = sanitize_capture(fixture)
    second_sanitized = SanitationGate(
        fixture.settings.capture,
        fixture.settings.sanitation,
    ).sanitize(validated, state_root / "sanitized")

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(
            publishers[0].publish,
            validated,
            first_sanitized,
        )
        second_future = executor.submit(
            publishers[1].publish,
            validated,
            second_sanitized,
        )
        published = (first_future.result(), second_future.result())

    assert published[0].manifest.bundle_id == published[1].manifest.bundle_id
    assert len(tuple((state_root / "bundles").iterdir())) == 1
    assert len(tuple((state_root / "runs").glob("*/current.json"))) == 1


def test_concurrent_superseding_publishers_serialize_retention(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    state_root = fixture.workspace / fixture.settings.capture.state_path
    first_publisher = RunBundlePublisher(
        state_root,
        fixture.settings.capture,
        fixture.settings.sanitation,
    )
    pipeline(fixture, first_publisher)
    fixture.save_checkpoint("completed")
    fixture.request = replace(
        fixture.request,
        completion_state=CompletionState.COMPLETE,
    )
    validated, first_sanitized = sanitize_capture(fixture)
    second_sanitized = SanitationGate(
        fixture.settings.capture,
        fixture.settings.sanitation,
    ).sanitize(validated, state_root / "sanitized")
    second_publisher = RunBundlePublisher(
        state_root,
        fixture.settings.capture,
        fixture.settings.sanitation,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(
            first_publisher.publish,
            validated,
            first_sanitized,
        )
        second_future = executor.submit(
            second_publisher.publish,
            validated,
            second_sanitized,
        )
        published = (first_future.result(), second_future.result())

    assert published[0].manifest.bundle_id == published[1].manifest.bundle_id
    retained = tuple(
        (fixture.workspace / fixture.settings.capture.quarantine_path).glob(
            "runs/*/generation-*"
        )
    )
    assert len(retained) == fixture.settings.capture.quarantine_retention_generations


def test_publish_fails_if_run_directory_detaches_during_marker_commit(
    tmp_path,
    monkeypatch,
):
    fixture = make_capture_fixture(tmp_path)
    state_root = fixture.workspace / fixture.settings.capture.state_path
    publisher = RunBundlePublisher(
        state_root,
        fixture.settings.capture,
        fixture.settings.sanitation,
    )
    validated, sanitized = sanitize_capture(fixture)
    real_write_atomic = bundle_module._write_atomic_file_at
    detached_run = state_root / "detached-run"

    def detach_run_then_write(
        parent_descriptor,
        name,
        payload,
        *,
        mode,
        maximum_staging_bytes,
    ):
        if name == "current.json":
            live_run = next((state_root / "runs").iterdir())
            live_run.rename(detached_run)
            live_run.mkdir(mode=0o700)
        real_write_atomic(
            parent_descriptor,
            name,
            payload,
            mode=mode,
            maximum_staging_bytes=maximum_staging_bytes,
        )

    monkeypatch.setattr(
        bundle_module,
        "_write_atomic_file_at",
        detach_run_then_write,
    )

    with pytest.raises(RunBundlePublicationError, match="run directory binding"):
        publisher.publish(validated, sanitized)

    assert (detached_run / "current.json").is_file()
    assert not next((state_root / "runs").iterdir()).joinpath("current.json").exists()


@pytest.mark.parametrize("failure_stage", ("object", "bundle", "current"))
def test_retry_recovers_fixed_publication_staging(
    tmp_path,
    monkeypatch,
    failure_stage,
):
    fixture = make_capture_fixture(tmp_path)
    state_root = fixture.workspace / fixture.settings.capture.state_path
    publisher = RunBundlePublisher(
        state_root,
        fixture.settings.capture,
        fixture.settings.sanitation,
    )
    validated, sanitized = sanitize_capture(fixture)
    real_replace = os.replace
    real_rename = os.rename
    failure_injected = False

    def fail_selected_replace(
        source,
        destination,
        *,
        src_dir_fd=None,
        dst_dir_fd=None,
    ):
        nonlocal failure_injected
        is_current = source == ".current.json.tmp"
        should_fail = (failure_stage == "current" and is_current) or (
            failure_stage == "object" and not is_current
        )
        if should_fail and not failure_injected:
            failure_injected = True
            raise OSError("injected publication interruption")
        real_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    def fail_selected_rename(
        source,
        destination,
        *,
        src_dir_fd=None,
        dst_dir_fd=None,
    ):
        nonlocal failure_injected
        if failure_stage == "bundle" and not failure_injected:
            failure_injected = True
            raise OSError("injected publication interruption")
        real_rename(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(os, "replace", fail_selected_replace)
    monkeypatch.setattr(os, "rename", fail_selected_rename)
    with pytest.raises(OSError, match="injected publication interruption"):
        publisher.publish(validated, sanitized)
    assert failure_injected

    monkeypatch.setattr(os, "replace", real_replace)
    monkeypatch.setattr(os, "rename", real_rename)
    published = publisher.publish(validated, sanitized)

    assert publisher.store.require_exact(published.manifest.bundle_id)
    assert not tuple((state_root / "objects" / "sha256").glob(".*.tmp"))
    assert not tuple((state_root / "bundles").glob(".bundle.*.tmp"))
    assert not tuple((state_root / "runs").glob("*/.current.json.tmp"))


def test_newer_capture_reclaims_interrupted_current_staging(
    tmp_path,
    monkeypatch,
):
    fixture = make_capture_fixture(tmp_path)
    state_root = fixture.workspace / fixture.settings.capture.state_path
    publisher = RunBundlePublisher(
        state_root,
        fixture.settings.capture,
        fixture.settings.sanitation,
    )
    _, first = pipeline(fixture, publisher)
    fixture.save_checkpoint("completed")
    fixture.request = replace(
        fixture.request,
        completion_state=CompletionState.COMPLETE,
    )
    interrupted_capture, interrupted_sanitized = sanitize_capture(fixture)
    real_replace = os.replace

    def interrupt_current(
        source,
        destination,
        *,
        src_dir_fd=None,
        dst_dir_fd=None,
    ):
        if source == ".current.json.tmp":
            raise OSError("injected current interruption")
        real_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(os, "replace", interrupt_current)
    with pytest.raises(OSError, match="injected current interruption"):
        publisher.publish(interrupted_capture, interrupted_sanitized)

    monkeypatch.setattr(os, "replace", real_replace)
    (fixture.workspace / "progress.txt").write_text(
        "new safe observation\n",
        encoding="utf-8",
    )
    fixture.request = replace(
        fixture.request,
        run_log_paths=("progress.txt",),
    )
    newer_capture, newer_sanitized = sanitize_capture(fixture)
    published = publisher.publish(newer_capture, newer_sanitized)

    assert published.manifest.capture_generation == 2
    assert published.manifest.supersedes_bundle_id == first.manifest.bundle_id
    assert not tuple((state_root / "runs").glob("*/.current.json.tmp"))


def test_later_capture_supersedes_bundle_and_prunes_old_quarantine(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    publisher = RunBundlePublisher(
        fixture.workspace / fixture.settings.capture.state_path,
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


def test_retention_preserves_exporter_uncommitted_generation_recovery(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    state_root = fixture.workspace / fixture.settings.capture.state_path
    publisher = RunBundlePublisher(
        state_root,
        fixture.settings.capture,
        fixture.settings.sanitation,
    )
    validated, first = pipeline(fixture, publisher)
    committed_path = validated.path
    uncommitted_path = committed_path.with_name("generation-00000000000000000001")
    shutil.copytree(committed_path, uncommitted_path)
    sanitized = SanitationGate(
        fixture.settings.capture,
        fixture.settings.sanitation,
    ).sanitize(validated, state_root / "sanitized")

    repeated = publisher.publish(validated, sanitized)

    assert repeated.manifest.bundle_id == first.manifest.bundle_id
    assert committed_path.is_dir()
    assert uncommitted_path.is_dir()
    recovered = RunCaptureExporter(
        fixture.settings.capture,
        fixture.settings.sanitation,
    ).export(fixture.request)
    assert recovered.path == committed_path
    assert not uncommitted_path.exists()


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
        fixture.workspace / fixture.settings.capture.state_path,
        fixture.settings.capture,
        fixture.settings.sanitation,
    )
    _, first = pipeline(fixture, publisher)
    manifest_values = first.manifest.to_dict()
    manifest_values.pop("bundle_id")
    manifest_values[field] = value
    wrong_bundle = RunBundle.mint(**manifest_values)
    wrong_bundle_path = (
        publisher.root / "bundles" / wrong_bundle.bundle_id.rsplit(":", 1)[1]
    )
    wrong_bundle_path.mkdir(mode=0o700)
    wrong_manifest_path = wrong_bundle_path / "manifest.json"
    wrong_manifest_path.write_bytes(wrong_bundle.to_json_bytes())
    wrong_manifest_path.chmod(0o400)
    wrong_bundle_path.chmod(0o500)
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
        pipeline(fixture, publisher)


def test_bundle_contract_binds_kapso_commit_to_artifact_environment(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    publisher = RunBundlePublisher(
        fixture.workspace / fixture.settings.capture.state_path,
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
        fixture.workspace / fixture.settings.capture.state_path,
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
        fixture.workspace / fixture.settings.capture.state_path,
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
        fixture.workspace / fixture.settings.capture.state_path,
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
        fixture.workspace / fixture.settings.capture.state_path,
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
        fixture.workspace / fixture.settings.capture.state_path,
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
        fixture.workspace / fixture.settings.capture.state_path,
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
