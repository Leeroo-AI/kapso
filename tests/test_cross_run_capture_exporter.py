from dataclasses import replace

import pytest

import kapso.cross_run.capture.exporter as exporter_module
from kapso.cross_run.canonical import (
    canonical_json_bytes,
    parse_json_bytes,
    tree_or_blob_digest,
)
from kapso.cross_run.capture.exporter import (
    CaptureDescriptor,
    CaptureExportError,
    RunCaptureExporter,
)
from kapso.cross_run.capture.journal import ExecutionRevisionEvent
from kapso.cross_run.capture.validator import CaptureValidator
from kapso.cross_run.contracts import CaptureManifest, CompletionState
from kapso.cross_run.github.command import GitHubCommandError
from cross_run_capture_fixtures import make_capture_fixture


def test_export_is_atomic_restricted_and_idempotent(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    exporter = RunCaptureExporter(
        fixture.settings.capture,
        fixture.settings.sanitation,
    )

    first = exporter.export(fixture.request)
    second = exporter.export(fixture.request)

    assert second.manifest.capture_manifest_id == first.manifest.capture_manifest_id
    assert first.path.stat().st_mode & 0o077 == 0
    assert first.manifest.capture_watermarks["execution_journal_event_count"] == 1
    assert first.descriptor.branch_snapshot_refs


def test_exporter_enforces_configured_git_output_bound(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    bounded_settings = replace(
        fixture.settings.capture,
        git_command_output_bytes=8,
    )

    with pytest.raises(GitHubCommandError, match="stdout exceeds configured limit"):
        RunCaptureExporter(
            bounded_settings,
            fixture.settings.sanitation,
        ).export(fixture.request)


def test_same_bytes_under_changed_configuration_publish_a_successor(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    exporter = RunCaptureExporter(
        fixture.settings.capture,
        fixture.settings.sanitation,
    )
    first = exporter.export(fixture.request)
    changed = replace(
        fixture.request,
        configuration_fingerprint="sha256:" + "f" * 64,
    )

    second = exporter.export(changed)

    assert second.manifest.capture_generation == 1
    assert second.manifest.capture_manifest_id != first.manifest.capture_manifest_id
    assert (
        second.manifest.configuration_fingerprint == changed.configuration_fingerprint
    )


def test_current_capture_cannot_cross_run_lineage(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    exporter = RunCaptureExporter(
        fixture.settings.capture,
        fixture.settings.sanitation,
    )
    first = exporter.export(fixture.request)
    wrong_run = "run_" + "b" * 32
    descriptor_values = first.descriptor.to_dict()
    descriptor_values["run_id"] = wrong_run
    wrong_descriptor = CaptureDescriptor.from_dict(descriptor_values)
    descriptor_path = first.path / first.manifest.artifact_refs["capture_descriptor"]
    descriptor_path.write_bytes(wrong_descriptor.to_json_bytes())
    descriptor_path.chmod(0o600)
    manifest_values = first.manifest.to_dict()
    manifest_values.pop("capture_manifest_id")
    manifest_values["run_id"] = wrong_run
    manifest_values["checksums"][first.manifest.artifact_refs["capture_descriptor"]] = (
        tree_or_blob_digest(wrong_descriptor.to_json_bytes())
    )
    wrong_manifest = CaptureManifest.mint(**manifest_values)
    manifest_path = first.path / "capture_manifest.json"
    manifest_path.write_bytes(wrong_manifest.to_json_bytes())
    manifest_path.chmod(0o600)
    marker_path = first.path.parent / "current.json"
    marker_path.write_bytes(
        canonical_json_bytes(
            {
                "capture_manifest_id": wrong_manifest.capture_manifest_id,
                "generation": wrong_manifest.capture_generation,
                "path": first.path.name,
            }
        )
        + b"\n"
    )
    marker_path.chmod(0o600)

    with pytest.raises(CaptureExportError, match="another run identity"):
        exporter.export(fixture.request)


def test_stopped_capture_materializes_checkpoint_limited_global_prefix(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    checkpoint_node = fixture.strategy.node_history[0]
    recovered = replace(checkpoint_node, execution_revision=1, score=0.9)
    recovered.evaluation_attempts = [
        replace(attempt, score=0.9, metrics={"quality": 0.9})
        for attempt in checkpoint_node.evaluation_attempts
    ]
    recovered.metrics = {"quality": 0.9}
    fixture.store.add_experiment(recovered)
    exporter = RunCaptureExporter(
        fixture.settings.capture,
        fixture.settings.sanitation,
    )

    captured = exporter.export(fixture.request)

    history_ref = captured.descriptor.artifact_refs["experiment_history"]
    captured_history = parse_json_bytes((captured.path / history_ref).read_bytes())
    journal_ref = captured.descriptor.artifact_refs["execution_event_journal"]
    assert captured_history["revision"] == 1
    assert captured_history["records"][0]["execution_revision"] == 0
    assert len((captured.path / journal_ref).read_bytes().splitlines()) == 1


def test_interrupted_successor_preserves_prior_generation_marker(tmp_path, monkeypatch):
    fixture = make_capture_fixture(tmp_path)
    exporter = RunCaptureExporter(
        fixture.settings.capture,
        fixture.settings.sanitation,
    )
    first = exporter.export(fixture.request)
    fixture.save_checkpoint("completed")
    completed = replace(fixture.request, completion_state=CompletionState.COMPLETE)
    real_replace = exporter_module.os.replace

    def interrupt_generation(source, destination):
        if str(destination).endswith("generation-00000000000000000001"):
            raise OSError("simulated generation interruption")
        real_replace(source, destination)

    monkeypatch.setattr(exporter_module.os, "replace", interrupt_generation)
    with pytest.raises(OSError, match="interruption"):
        exporter.export(completed)
    marker = first.path.parent / "current.json"
    assert first.manifest.capture_manifest_id in marker.read_text(encoding="utf-8")


def test_orphan_generation_after_marker_crash_is_recovered(tmp_path, monkeypatch):
    fixture = make_capture_fixture(tmp_path)
    exporter = RunCaptureExporter(
        fixture.settings.capture,
        fixture.settings.sanitation,
    )
    real_marker_write = exporter_module._write_atomic_json

    def interrupt_marker(path, payload):
        raise OSError("simulated marker interruption")

    monkeypatch.setattr(exporter_module, "_write_atomic_json", interrupt_marker)
    with pytest.raises(OSError, match="marker interruption"):
        exporter.export(fixture.request)
    monkeypatch.setattr(exporter_module, "_write_atomic_json", real_marker_write)

    recovered = exporter.export(fixture.request)

    assert recovered.manifest.capture_generation == 0
    assert (recovered.path.parent / "current.json").is_file()


def test_export_requires_explicit_full_evaluator_fingerprint(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    request = replace(fixture.request, evaluation_fingerprints=())

    with pytest.raises(CaptureExportError, match="full fingerprint"):
        RunCaptureExporter(
            fixture.settings.capture,
            fixture.settings.sanitation,
        ).export(request)


def test_export_rejects_duplicate_seed_attempts_before_aggregation(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    original = fixture.strategy.node_history[0]
    attempt = original.evaluation_attempts[0]
    recovered = replace(original, execution_revision=1, score=0.5)
    recovered.evaluation_attempts = [
        replace(attempt, score=0.2, metrics={"quality": 0.2}),
        replace(attempt, score=0.8, metrics={"quality": 0.8}),
    ]
    recovered.metrics = {"quality": 0.5}
    fixture.store.add_experiment(recovered)
    fixture.strategy.node_history = [recovered]
    fixture.save_checkpoint("running")

    with pytest.raises(CaptureExportError, match="duplicate seed"):
        RunCaptureExporter(
            fixture.settings.capture,
            fixture.settings.sanitation,
        ).export(fixture.request)


def test_export_binds_fingerprint_metric_to_score_evidence(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    fingerprint = fixture.request.evaluation_fingerprints[0]
    values = fingerprint.to_dict()
    values.pop("evaluation_fingerprint_id")
    values["metric_name"] = "wrong_metric"
    wrong_metric = type(fingerprint).mint(**values)
    request = replace(fixture.request, evaluation_fingerprints=(wrong_metric,))

    with pytest.raises(CaptureExportError, match="full fingerprint"):
        RunCaptureExporter(
            fixture.settings.capture,
            fixture.settings.sanitation,
        ).export(request)


def test_export_rejects_self_consistent_git_refs_that_forge_the_idea_parent(
    tmp_path,
):
    fixture = make_capture_fixture(tmp_path)
    node = fixture.strategy.node_history[0]
    candidate_commit = node.evaluation_attempts[0].commit_sha
    node.implementation_base_ref = candidate_commit
    node.diff_base_ref = candidate_commit
    node.feedback_base_ref = candidate_commit
    fixture.save_checkpoint("running")

    journal_path = fixture.store.revision_journal.path
    event_data = parse_json_bytes(journal_path.read_bytes().strip())
    event_data.pop("event_id")
    for name in ("implementation", "diff", "feedback"):
        event_data["artifact_refs"][f"{name}_base"] = candidate_commit
        event_data["artifact_refs"][f"{name}_base_commit"] = candidate_commit
    forged_event = ExecutionRevisionEvent.mint(**event_data)
    journal_path.write_bytes(canonical_json_bytes(forged_event.to_dict()) + b"\n")

    with pytest.raises(CaptureExportError, match="artifact provenance is not exact"):
        RunCaptureExporter(
            fixture.settings.capture,
            fixture.settings.sanitation,
        ).export(fixture.request)


def test_export_rejects_secondary_metric_that_duplicates_the_primary_score(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    original = fixture.strategy.node_history[0]
    revised = replace(original, execution_revision=1)
    revised.evaluation_attempts = [
        replace(
            attempt,
            metrics={**attempt.metrics, "secondary": attempt.score},
        )
        for attempt in original.evaluation_attempts
    ]
    revised.metrics = {**original.metrics, "secondary": original.score}
    fixture.store.add_experiment(revised)
    fixture.strategy.node_history = [revised]
    fixture.save_checkpoint("running")
    fingerprint = fixture.request.evaluation_fingerprints[0]
    values = fingerprint.to_dict()
    values.pop("evaluation_fingerprint_id")
    values["metric_name"] = "secondary"
    wrong_metric = type(fingerprint).mint(**values)
    request = replace(fixture.request, evaluation_fingerprints=(wrong_metric,))

    with pytest.raises(CaptureExportError, match="full fingerprint"):
        RunCaptureExporter(
            fixture.settings.capture,
            fixture.settings.sanitation,
        ).export(request)


def test_export_rejects_symlinked_or_denied_run_logs(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    (fixture.workspace / "run.txt").symlink_to(outside)
    exporter = RunCaptureExporter(
        fixture.settings.capture,
        fixture.settings.sanitation,
    )

    with pytest.raises(CaptureExportError, match="symlink"):
        exporter.export(replace(fixture.request, run_log_paths=("run.txt",)))
    with pytest.raises(CaptureExportError, match="denied"):
        exporter.export(
            replace(fixture.request, run_log_paths=("logs/credentials.json",))
        )


def test_each_recovery_revision_keeps_its_immutable_git_commit(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    original = fixture.strategy.node_history[0]
    original_commit = original.evaluation_attempts[0].commit_sha
    repo = fixture.strategy.workspace.repo
    repo.git.checkout(original.branch_name)
    (fixture.workspace / "solution.py").write_text("VALUE = 2\n", encoding="utf-8")
    repo.git.add(["solution.py"])
    repo.git.commit("-m", "recovered candidate")
    recovered_commit = repo.head.commit.hexsha
    repo.git.checkout("main")
    recovered = replace(original, execution_revision=1, score=0.9)
    recovered.implementation_base_ref = original_commit
    recovered.evaluation_attempts = [
        replace(
            original.evaluation_attempts[0],
            commit_sha=recovered_commit,
            score=0.9,
            metrics={"quality": 0.9},
        )
    ]
    recovered.metrics = {"quality": 0.9}
    fixture.store.add_experiment(recovered)
    fixture.strategy.node_history = [recovered]
    fixture.save_checkpoint("running")

    exported = RunCaptureExporter(
        fixture.settings.capture,
        fixture.settings.sanitation,
    ).export(fixture.request)

    commits = {
        branch.execution_revision: branch.commit_sha
        for branch in CaptureValidator().validate(exported.path).branch_snapshots
    }
    assert commits == {0: original_commit, 1: recovered_commit}
