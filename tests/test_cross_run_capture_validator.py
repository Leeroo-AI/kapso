import pytest

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    parse_json_bytes,
    tree_or_blob_digest,
)
from kapso.cross_run.capture.exporter import RunCaptureExporter
from kapso.cross_run.capture.validator import CaptureValidationError, CaptureValidator
from kapso.cross_run.contracts import CaptureManifest
from cross_run_capture_fixtures import make_capture_fixture


def replace_payload_and_remint_manifest(exported, relative_path, document):
    payload = canonical_json_bytes(document)
    payload_path = exported.path / relative_path
    payload_path.write_bytes(payload)
    payload_path.chmod(0o600)
    values = exported.manifest.to_dict()
    values.pop("capture_manifest_id")
    values["checksums"][relative_path] = tree_or_blob_digest(payload)
    manifest = CaptureManifest.mint(**values)
    manifest_path = exported.path / "capture_manifest.json"
    manifest_path.write_bytes(manifest.to_json_bytes())
    manifest_path.chmod(0o600)


def replace_bytes_and_remint_manifest(exported, relative_path, payload):
    payload_path = exported.path / relative_path
    payload_path.write_bytes(payload)
    payload_path.chmod(0o600)
    values = exported.manifest.to_dict()
    values.pop("capture_manifest_id")
    values["checksums"][relative_path] = tree_or_blob_digest(payload)
    manifest = CaptureManifest.mint(**values)
    manifest_path = exported.path / "capture_manifest.json"
    manifest_path.write_bytes(manifest.to_json_bytes())
    manifest_path.chmod(0o600)


def test_validator_proves_full_cross_artifact_frontier(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    exported = RunCaptureExporter(
        fixture.settings.capture,
        fixture.settings.sanitation,
    ).export(fixture.request)

    validated = CaptureValidator().validate(exported.path)

    assert (
        validated.history.experiments[0].idea_id == validated.archive.ideas[0].idea_id
    )
    assert validated.branch_snapshots[0].evaluated_commit_shas


def test_validator_rejects_any_payload_tampering(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    exported = RunCaptureExporter(
        fixture.settings.capture,
        fixture.settings.sanitation,
    ).export(fixture.request)
    history_path = (
        exported.path / exported.descriptor.artifact_refs["experiment_history"]
    )
    history_path.write_bytes(history_path.read_bytes() + b" ")

    with pytest.raises(CaptureValidationError, match="checksum mismatch"):
        CaptureValidator().validate(exported.path)


def test_validator_rejects_unknown_checkpoint_fields_after_valid_remint(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    exported = RunCaptureExporter(
        fixture.settings.capture,
        fixture.settings.sanitation,
    ).export(fixture.request)
    checkpoint_ref = exported.descriptor.artifact_refs["checkpoint"]
    checkpoint = parse_json_bytes((exported.path / checkpoint_ref).read_bytes())
    checkpoint["unknown"] = "not part of the authority schema"
    replace_payload_and_remint_manifest(exported, checkpoint_ref, checkpoint)

    with pytest.raises(CaptureValidationError, match="checkpoint schema is not exact"):
        CaptureValidator().validate(exported.path)


def test_validator_rejects_defaulted_checkpoint_node_fields(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    exported = RunCaptureExporter(
        fixture.settings.capture,
        fixture.settings.sanitation,
    ).export(fixture.request)
    checkpoint_ref = exported.descriptor.artifact_refs["checkpoint"]
    checkpoint = parse_json_bytes((exported.path / checkpoint_ref).read_bytes())
    checkpoint["strategy_state"]["node_history"][0].pop("agent_output")
    replace_payload_and_remint_manifest(exported, checkpoint_ref, checkpoint)

    with pytest.raises(CaptureValidationError, match="node schema is not exact"):
        CaptureValidator().validate(exported.path)


def test_validator_recomputes_raw_git_commit_identity(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    exported = RunCaptureExporter(
        fixture.settings.capture,
        fixture.settings.sanitation,
    ).export(fixture.request)
    branch = parse_json_bytes(
        (exported.path / exported.descriptor.branch_snapshot_refs[0]).read_bytes()
    )
    commit_ref = branch["commit_objects"][0]["payload_ref"]
    payload = (exported.path / commit_ref).read_bytes() + b"tampered\n"
    replace_bytes_and_remint_manifest(exported, commit_ref, payload)

    with pytest.raises(CaptureValidationError, match="commit payload identity"):
        CaptureValidator().validate(exported.path)


def test_validator_requires_complete_excluded_git_tree_partition(tmp_path):
    fixture = make_capture_fixture(tmp_path, forbidden_artifacts=True)
    exported = RunCaptureExporter(
        fixture.settings.capture,
        fixture.settings.sanitation,
    ).export(fixture.request)
    branch_ref = exported.descriptor.branch_snapshot_refs[0]
    branch = parse_json_bytes((exported.path / branch_ref).read_bytes())
    branch["excluded_files"].pop()
    replace_payload_and_remint_manifest(exported, branch_ref, branch)

    with pytest.raises(CaptureValidationError, match="root tree proof"):
        CaptureValidator().validate(exported.path)


def test_validator_rejects_archive_parent_ref_forged_to_evaluated_commit(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    exported = RunCaptureExporter(
        fixture.settings.capture,
        fixture.settings.sanitation,
    ).export(fixture.request)
    archive_ref = exported.descriptor.artifact_refs["idea_archive"]
    archive = parse_json_bytes((exported.path / archive_ref).read_bytes())
    branch = parse_json_bytes(
        (exported.path / exported.descriptor.branch_snapshot_refs[0]).read_bytes()
    )
    resolved_parent = archive["ideas"][0]["resolved_parent"]
    for name in (
        "git_ref",
        "materialized_ref",
        "diff_base_ref",
        "feedback_base_ref",
    ):
        resolved_parent[name] = branch["commit_sha"]
    replace_payload_and_remint_manifest(exported, archive_ref, archive)

    with pytest.raises(
        CaptureValidationError,
        match="artifact provenance is not exact",
    ):
        CaptureValidator().validate(exported.path)
