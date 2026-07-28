"""The shared capture CAS retains exact local expert replay evidence."""

from __future__ import annotations

import time

import pytest

from kapso.cross_run.capture.bundle import (
    RunBundlePublicationError,
    RunBundleStore,
    StoredSourceReplayContextProvider,
)
from kapso.cross_run.capture.pipeline import RunCaptureContext, RunCapturePipeline
from kapso.cross_run.expert.task_evaluation_materialization import (
    TaskEvaluationMaterializationLimits,
)
from kapso.cross_run.launch.starting_artifacts import (
    build_launch_starting_artifact_provider,
)
from kapso.cross_run.contracts import TaskContextBinding
from cross_run_capture_fixtures import make_capture_fixture
from test_launch_resolver import resolver_case


def test_shared_store_imports_bundle_and_launch_artifacts_exactly(
    tmp_path,
    resolver_case,
):
    run_root = tmp_path / "run"
    run_root.mkdir()
    capture = make_capture_fixture(run_root)
    stored = RunCapturePipeline(
        RunCaptureContext(capture.request),
        capture.settings,
    ).capture_if_due(capture.request.completion_state, force=True)
    assert stored is not None
    state_root = (tmp_path / "shared").resolve()
    state_root.mkdir(mode=0o700)
    store = RunBundleStore.initialize(
        state_root / capture.settings.capture.state_path,
        capture.settings.capture,
        capture.settings.sanitation,
    )
    task_context, artifact_provider = _starting_context(
        resolver_case,
        tmp_path,
        capture,
    )

    imported = store.import_exact(stored)
    context = store.publish_starting_artifacts(
        task_context_binding=task_context,
        launch_artifacts=artifact_provider.artifacts,
        validation_settings=capture.settings.expert.validation,
    )
    reopened = RunBundleStore(
        store.root,
        capture.settings.capture,
        capture.settings.sanitation,
    )
    expected_ids = {
        item.artifact.starting_artifact_ref: (
            item.artifact.starting_artifact_content_id
        )
        for item in context.starting_artifacts
    }
    rematerialized = StoredSourceReplayContextProvider(
        reopened,
        capture.settings.expert.validation,
    ).materialize_exact(
        task_context,
        expected_ids,
        _limits(capture),
    )

    assert imported == stored
    assert reopened.read_exact(stored.manifest.bundle_id) == stored
    assert rematerialized == context
    assert all(
        artifact_id.startswith("source-replay-starting-artifact:sha256:")
        for artifact_id in expected_ids.values()
    )


def test_shared_store_rejects_substitution_and_bounded_overread(
    tmp_path,
    resolver_case,
):
    run_root = tmp_path / "run"
    run_root.mkdir()
    capture = make_capture_fixture(run_root)
    stored = RunCapturePipeline(
        RunCaptureContext(capture.request),
        capture.settings,
    ).capture_if_due(capture.request.completion_state, force=True)
    assert stored is not None
    state_root = (tmp_path / "shared").resolve()
    state_root.mkdir(mode=0o700)
    store = RunBundleStore.initialize(
        state_root / capture.settings.capture.state_path,
        capture.settings.capture,
        capture.settings.sanitation,
    )
    store.import_exact(stored)
    task_context, artifact_provider = _starting_context(
        resolver_case,
        tmp_path,
        capture,
    )
    context = store.publish_starting_artifacts(
        task_context_binding=task_context,
        launch_artifacts=artifact_provider.artifacts,
        validation_settings=capture.settings.expert.validation,
    )
    expected_ids = {
        item.artifact.starting_artifact_ref: (
            item.artifact.starting_artifact_content_id
        )
        for item in context.starting_artifacts
    }

    with pytest.raises(RunBundlePublicationError, match="task context"):
        store.materialize_exact(
            task_context,
            {},
            _limits(capture),
            validation_settings=capture.settings.expert.validation,
        )
    with pytest.raises(RunBundlePublicationError, match="materialization budget"):
        store.read_exact_bounded(
            stored.manifest.bundle_id,
            maximum_entries=capture.settings.capture.bundle_entry_limit,
            maximum_bytes=1,
            deadline=time.monotonic() + 30,
        )
    with pytest.raises(RunBundlePublicationError, match="byte budget"):
        store.materialize_exact(
            task_context,
            expected_ids,
            TaskEvaluationMaterializationLimits(
                maximum_entries=_limits(capture).maximum_entries,
                maximum_bytes=1,
                timeout_seconds=_limits(capture).timeout_seconds,
            ),
            validation_settings=capture.settings.expert.validation,
        )


def _limits(capture) -> TaskEvaluationMaterializationLimits:
    policy = capture.settings.expert.validation.policy
    return TaskEvaluationMaterializationLimits(
        maximum_entries=policy.task_evaluation_materialization_entry_limit,
        maximum_bytes=policy.task_evaluation_materialization_byte_limit,
        timeout_seconds=policy.task_evaluation_materialization_timeout_seconds,
    )


def _starting_context(resolver_case, tmp_path, capture):
    source = (tmp_path / "task-data").resolve()
    source.mkdir()
    (source / "input.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    provider = build_launch_starting_artifact_provider(
        sources={"task-data": (source, "kapso_datasets")},
        settings=capture.settings.launch,
    )
    launch = resolver_case["request"]
    original = launch.task_context_request.bind(
        binding=launch.binding,
        scope_contract=resolver_case["evidence"].validation_context.scope_contract,
    )
    context = TaskContextBinding.mint(
        scope_contract_id=original.scope_contract_id,
        scope_id=original.scope_id,
        task_family_id=original.task_family_id,
        task_adapter_id=original.task_adapter_id,
        capability_tags=original.capability_tags,
        input_contract_fingerprint=original.input_contract_fingerprint,
        target_contract_fingerprint=original.target_contract_fingerprint,
        starting_artifact_refs=("task-data",),
        method_fingerprint=original.method_fingerprint,
        toolchain_fingerprint=original.toolchain_fingerprint,
        dependency_runtime_fingerprint=original.dependency_runtime_fingerprint,
        budget_hardware_envelope=original.budget_hardware_envelope,
        transfer_dimensions=original.transfer_dimensions,
    )
    return context, provider
