"""Synchronous production seam from durable checkpoints to local RunBundles."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from time import monotonic

from kapso.cross_run.capture.bundle import RunBundlePublisher, StoredRunBundle
from kapso.cross_run.capture.exporter import (
    RunCaptureExporter,
    RunCaptureRequest,
)
from kapso.cross_run.capture.sanitation import SanitationGate
from kapso.cross_run.capture.validator import CaptureValidator
from kapso.cross_run.contracts import CompletionState
from kapso.cross_run.settings import CrossRunSettings
from kapso.execution.memories.experiment_memory.store import ExperimentHistoryStore


@dataclass(frozen=True)
class RunCaptureContext:
    """Complete externally verified identity needed to capture one run."""

    request_template: RunCaptureRequest

    def __post_init__(self) -> None:
        if self.request_template.completion_state is not CompletionState.STOPPED:
            raise ValueError("capture context template must use stopped state")

    def request(self, completion_state: CompletionState) -> RunCaptureRequest:
        return replace(self.request_template, completion_state=completion_state)


class RunCapturePipeline:
    """Capture, validate, sanitize, and publish one proven durable frontier."""

    def __init__(
        self,
        context: RunCaptureContext,
        settings: CrossRunSettings,
    ):
        self.context = context
        self.settings = settings
        self.exporter = RunCaptureExporter(settings.capture, settings.sanitation)
        self.validator = CaptureValidator(settings.capture.score_comparison_tolerance)
        self.sanitation_gate = SanitationGate(
            settings.capture,
            settings.sanitation,
        )
        state_root = (
            context.request_template.workspace_dir / settings.capture.state_path
        )
        self.sanitized_root = state_root / "sanitized"
        self.publisher = RunBundlePublisher(
            state_root,
            settings.capture,
            settings.sanitation,
        )
        self.last_successful_capture_monotonic: float | None = None

    def validate_runtime_binding(
        self,
        workspace_dir: str | Path,
        experiment_store: ExperimentHistoryStore,
        idea_archive_path: str | Path,
    ) -> None:
        template = self.context.request_template
        if Path(workspace_dir).resolve() != template.workspace_dir.resolve():
            raise ValueError("capture pipeline workspace identity changed")
        if (
            experiment_store.run_id != template.run_id
            or experiment_store.campaign_id != template.campaign_id
        ):
            raise ValueError("capture pipeline run/campaign identity changed")
        if Path(idea_archive_path).resolve() != template.idea_archive_path.resolve():
            raise ValueError("capture pipeline idea archive identity changed")

    def capture_if_due(
        self,
        completion_state: CompletionState,
        *,
        force: bool = False,
    ) -> StoredRunBundle | None:
        now = monotonic()
        if (
            not force
            and self.last_successful_capture_monotonic is not None
            and now - self.last_successful_capture_monotonic
            < self.settings.capture.capture_interval_seconds
        ):
            return None
        exported = self.exporter.export(self.context.request(completion_state))
        validated = self.validator.validate(exported.path)
        sanitized = self.sanitation_gate.sanitize(
            validated,
            self.sanitized_root,
        )
        stored = self.publisher.publish(validated, sanitized)
        self.last_successful_capture_monotonic = now
        return stored
