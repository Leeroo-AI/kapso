"""Exact RunBundle supersession lineage resolution."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from kapso.cross_run.canonical import content_id
from kapso.cross_run.capture.bundle import RunBundleStore
from kapso.cross_run.capture.bundle_lineage import validate_run_bundle_successor
from kapso.cross_run.capture.pipeline import RunCaptureContext, RunCapturePipeline
from kapso.cross_run.catalog.lineage import (
    RunBundleLineageError,
    RunBundleLineageProvider,
)
from kapso.cross_run.catalog.projector import RunBundleProjector
from kapso.cross_run.contracts import CompletionState, RunBundle
from cross_run_capture_fixtures import make_capture_fixture
from test_cross_run_contracts import build_records


@dataclass(frozen=True)
class _ManifestStub:
    bundle_id: str
    supersedes_bundle_id: str | None
    capture_generation: int


@dataclass(frozen=True)
class _ReaderStub:
    manifest: _ManifestStub


@dataclass(frozen=True)
class _ProjectionStub:
    source_bundle: _ManifestStub


class _RecordingSource:
    def __init__(self, readers):
        self.readers = readers
        self.requested_ids = []

    def read_exact(self, bundle_id):
        self.requested_ids.append(bundle_id)
        return self.readers.get(bundle_id)


class _SequencedSource:
    def __init__(self, responses):
        self.responses = responses
        self.request_counts = {}

    def read_exact(self, bundle_id):
        request_count = self.request_counts.get(bundle_id, 0)
        self.request_counts[bundle_id] = request_count + 1
        return self.responses[bundle_id][request_count]


class _RejectingProjector:
    @staticmethod
    def project(bundle, previous=None):
        raise AssertionError("invalid lineage reached projection")


class _RecordingProjector:
    def __init__(self):
        self.projected_ids = []

    def project(self, bundle, previous=None):
        self.projected_ids.append(bundle.manifest.bundle_id)
        return _ProjectionStub(source_bundle=bundle.manifest)


def _bundle_id(label: str) -> str:
    return content_id("run-bundle", {"label": label})


def _stub(
    bundle_id: str,
    predecessor_id: str | None,
    capture_generation: int = 0,
) -> _ReaderStub:
    return _ReaderStub(
        manifest=_ManifestStub(
            bundle_id=bundle_id,
            supersedes_bundle_id=predecessor_id,
            capture_generation=capture_generation,
        )
    )


def _remint_bundle(bundle: RunBundle, **changes) -> RunBundle:
    values = bundle.to_dict()
    values.pop("bundle_id")
    values.update(changes)
    return RunBundle.mint(**values)


def test_lineage_provider_replays_exact_root_to_tip_chain_once(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    pipeline = RunCapturePipeline(
        RunCaptureContext(fixture.request),
        fixture.settings,
    )
    first = pipeline.capture_if_due(CompletionState.STOPPED, force=True)
    fixture.strategy.previous_errors = ["first observed execution difficulty"]
    fixture.save_checkpoint("running")
    second = pipeline.capture_if_due(CompletionState.STOPPED, force=True)
    fixture.strategy.previous_errors.append("second observed execution difficulty")
    fixture.save_checkpoint("running")
    third = pipeline.capture_if_due(CompletionState.STOPPED, force=True)
    assert first is not None and second is not None and third is not None
    store = RunBundleStore(
        fixture.workspace / fixture.settings.capture.state_path,
        fixture.settings.capture,
        fixture.settings.sanitation,
    )
    source = _RecordingSource(
        {
            bundle.manifest.bundle_id: store.require_exact(bundle.manifest.bundle_id)
            for bundle in (first, second, third)
        }
    )
    provider = RunBundleLineageProvider(
        source,
        RunBundleProjector(fixture.settings.capture.score_comparison_tolerance),
        fixture.settings.capture.bundle_lineage_limit,
    )

    lineage = provider.resolve_exact(third.manifest.bundle_id)

    assert lineage.bundle_ids == (
        first.manifest.bundle_id,
        second.manifest.bundle_id,
        third.manifest.bundle_id,
    )
    assert source.requested_ids == [
        third.manifest.bundle_id,
        second.manifest.bundle_id,
        first.manifest.bundle_id,
        first.manifest.bundle_id,
        second.manifest.bundle_id,
        third.manifest.bundle_id,
    ]
    assert lineage.tip_bundle.manifest == third.manifest
    assert lineage.tip_projection.source_bundle == third.manifest
    assert lineage.tip_projection.episodes[0].supersedes_projection_id is not None


def test_lineage_provider_rejects_cycle_before_projection():
    tip_id = _bundle_id("cycle-tip")
    predecessor_id = _bundle_id("cycle-predecessor")
    source = _RecordingSource(
        {
            tip_id: _stub(tip_id, predecessor_id),
            predecessor_id: _stub(predecessor_id, tip_id),
        }
    )
    provider = RunBundleLineageProvider(source, _RejectingProjector(), 2)

    with pytest.raises(RunBundleLineageError, match="cycle"):
        provider.resolve_exact(tip_id)

    assert source.requested_ids == [tip_id, predecessor_id]


def test_lineage_provider_rejects_self_cycle_without_second_lookup():
    tip_id = _bundle_id("self-cycle")
    source = _RecordingSource({tip_id: _stub(tip_id, tip_id)})
    provider = RunBundleLineageProvider(source, _RejectingProjector(), 2)

    with pytest.raises(RunBundleLineageError, match="cycle"):
        provider.resolve_exact(tip_id)

    assert source.requested_ids == [tip_id]


def test_lineage_provider_accepts_chain_at_exact_configured_limit():
    tip_id = _bundle_id("exact-limit-tip")
    predecessor_id = _bundle_id("exact-limit-predecessor")
    root_id = _bundle_id("exact-limit-root")
    source = _RecordingSource(
        {
            tip_id: _stub(tip_id, predecessor_id, capture_generation=2),
            predecessor_id: _stub(
                predecessor_id,
                root_id,
                capture_generation=1,
            ),
            root_id: _stub(root_id, None),
        }
    )
    projector = _RecordingProjector()
    provider = RunBundleLineageProvider(source, projector, 3)

    lineage = provider.resolve_exact(tip_id)

    assert lineage.bundle_ids == (root_id, predecessor_id, tip_id)
    assert source.requested_ids == [
        tip_id,
        predecessor_id,
        root_id,
        root_id,
        predecessor_id,
        tip_id,
    ]
    assert projector.projected_ids == [root_id, predecessor_id, tip_id]


def test_lineage_provider_stops_at_bound_without_fetching_deeper_predecessor():
    tip_id = _bundle_id("bound-tip")
    predecessor_id = _bundle_id("bound-predecessor")
    root_id = _bundle_id("bound-root")
    source = _RecordingSource(
        {
            tip_id: _stub(tip_id, predecessor_id),
            predecessor_id: _stub(predecessor_id, root_id),
            root_id: _stub(root_id, None),
        }
    )
    provider = RunBundleLineageProvider(source, _RejectingProjector(), 2)

    with pytest.raises(RunBundleLineageError, match="configured depth"):
        provider.resolve_exact(tip_id)

    assert source.requested_ids == [tip_id, predecessor_id]


def test_lineage_provider_rejects_missing_exact_predecessor():
    tip_id = _bundle_id("missing-tip")
    missing_id = _bundle_id("missing-predecessor")
    source = _RecordingSource({tip_id: _stub(tip_id, missing_id)})
    provider = RunBundleLineageProvider(source, _RejectingProjector(), 2)

    with pytest.raises(RunBundleLineageError, match="missing exact predecessor"):
        provider.resolve_exact(tip_id)

    assert source.requested_ids == [tip_id, missing_id]


def test_lineage_provider_rejects_missing_requested_head():
    requested_id = _bundle_id("missing-head")
    source = _RecordingSource({})
    provider = RunBundleLineageProvider(source, _RejectingProjector(), 1)

    with pytest.raises(RunBundleLineageError, match=requested_id):
        provider.resolve_exact(requested_id)

    assert source.requested_ids == [requested_id]


def test_lineage_provider_rejects_source_identity_substitution():
    requested_id = _bundle_id("requested")
    substituted_id = _bundle_id("substituted")
    source = _RecordingSource({requested_id: _stub(substituted_id, None)})
    provider = RunBundleLineageProvider(source, _RejectingProjector(), 1)

    with pytest.raises(RunBundleLineageError, match="another bundle identity"):
        provider.resolve_exact(requested_id)


def test_lineage_provider_rejects_bundle_disappearing_before_projection():
    root_id = _bundle_id("disappearing-root")
    source = _SequencedSource({root_id: (_stub(root_id, None), None)})
    provider = RunBundleLineageProvider(source, _RejectingProjector(), 1)

    with pytest.raises(RunBundleLineageError, match="changed during projection"):
        provider.resolve_exact(root_id)


def test_lineage_provider_rejects_second_pass_identity_substitution():
    root_id = _bundle_id("stable-root")
    substituted_id = _bundle_id("second-pass-substitution")
    source = _SequencedSource(
        {
            root_id: (
                _stub(root_id, None),
                _stub(substituted_id, None),
            )
        }
    )
    provider = RunBundleLineageProvider(source, _RejectingProjector(), 1)

    with pytest.raises(RunBundleLineageError, match="changed identity"):
        provider.resolve_exact(root_id)


def test_lineage_provider_rejects_truncated_nonzero_generation_root():
    root_id = _bundle_id("forged-root")
    source = _RecordingSource({root_id: _stub(root_id, None, capture_generation=4)})
    provider = RunBundleLineageProvider(source, _RejectingProjector(), 1)

    with pytest.raises(RunBundleLineageError, match="root is not generation zero"):
        provider.resolve_exact(root_id)


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"capture_generation": 2}, "not contiguous"),
        ({"checkpoint_frontier": 0}, "frontier moved backwards"),
        ({"capture_watermarks": {"other": 1}}, "watermarks moved backwards"),
        ({"run_id": "another-run"}, "stable run identity"),
    ),
)
def test_shared_lineage_edge_rejects_semantic_splices(changes, message):
    root = next(record for record in build_records() if isinstance(record, RunBundle))
    successor_values = {
        "capture_generation": 1,
        "supersedes_bundle_id": root.bundle_id,
        **changes,
    }
    successor = _remint_bundle(root, **successor_values)

    with pytest.raises(ValueError, match=message):
        validate_run_bundle_successor(root, successor, ValueError)


@pytest.mark.parametrize("maximum_bundles", (0, -1, True, 1.5))
def test_lineage_provider_requires_positive_integer_bound(maximum_bundles):
    with pytest.raises(RunBundleLineageError, match="limit must be positive"):
        RunBundleLineageProvider(
            _RecordingSource({}),
            _RejectingProjector(),
            maximum_bundles,
        )
