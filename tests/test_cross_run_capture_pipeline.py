from dataclasses import replace

import pytest

import kapso.cross_run.capture.pipeline as pipeline_module
from kapso.cross_run.capture.pipeline import RunCaptureContext, RunCapturePipeline
from kapso.cross_run.contracts import CompletionState
from cross_run_capture_fixtures import make_capture_fixture


def test_pipeline_publishes_first_frontier_and_honors_interval(tmp_path, monkeypatch):
    fixture = make_capture_fixture(tmp_path)
    clock = iter((100.0, 101.0, 200.0))
    monkeypatch.setattr(pipeline_module, "monotonic", lambda: next(clock))
    pipeline = RunCapturePipeline(
        RunCaptureContext(fixture.request),
        fixture.settings,
    )

    first = pipeline.capture_if_due(CompletionState.STOPPED)
    skipped = pipeline.capture_if_due(CompletionState.STOPPED)
    forced = pipeline.capture_if_due(CompletionState.STOPPED, force=True)

    assert first is not None
    assert skipped is None
    assert forced is not None
    assert forced.manifest.bundle_id == first.manifest.bundle_id
    assert pipeline.publisher.load(first.manifest.bundle_id).manifest == first.manifest


def test_pipeline_rejects_runtime_identity_drift(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    wrong_request = replace(fixture.request, run_id="run_" + "b" * 32)
    pipeline = RunCapturePipeline(
        RunCaptureContext(wrong_request),
        fixture.settings,
    )

    with pytest.raises(ValueError, match="run/campaign identity"):
        pipeline.validate_runtime_binding(
            fixture.workspace,
            fixture.store,
            fixture.request.idea_archive_path,
        )
