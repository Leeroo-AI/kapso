"""Deterministic structural and provenance validation for raw captures."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    parse_json_bytes,
    to_json_value,
    tree_or_blob_digest,
)
from kapso.cross_run.capture.branch_evidence import validate_branch_evidence
from kapso.cross_run.capture.exporter import (
    CAPTURE_DESCRIPTOR_REF,
    CAPTURE_MANIFEST_FILENAME,
    BranchSnapshot,
    CaptureDescriptor,
)
from kapso.cross_run.capture.evaluation_evidence import (
    validate_evaluation_fingerprints,
)
from kapso.cross_run.record_contracts import ExecutionRevisionEvent
from kapso.cross_run.capture.provenance import validate_execution_provenance
from kapso.cross_run.contracts import (
    ArtifactCompleteness,
    CaptureManifest,
    CompletionState,
)
from kapso.execution.memories.experiment_memory.record import ExperimentRecord
from kapso.execution.memories.experiment_memory.store import ExperimentHistoryStore
from kapso.execution.run_checkpoint import RunCheckpoint
from kapso.execution.search_strategies.node import SearchNode
from kapso.execution.search_strategies.generic.strategy import (
    GENERIC_SEARCH_STATE_FIELDS,
    GENERIC_SEARCH_STATE_SCHEMA,
)
from kapso.execution.search_strategies.generic.ideation.archive import IdeaArchiveState


class CaptureValidationError(ValueError):
    """A capture is malformed or its authorities do not share one frontier."""


@dataclass(frozen=True)
class ValidatedCapture:
    path: Path
    directory_identity: tuple[int, int]
    manifest: CaptureManifest
    descriptor: CaptureDescriptor
    checkpoint: RunCheckpoint
    history: ExperimentHistoryStore
    events: tuple[ExecutionRevisionEvent, ...]
    archive: IdeaArchiveState
    nodes: tuple[SearchNode, ...]
    branch_snapshots: tuple[BranchSnapshot, ...]


class CaptureValidator:
    """Validate hashes, watermarks, joins, measurements, and Git provenance."""

    def __init__(self, score_comparison_tolerance: float):
        if score_comparison_tolerance <= 0.0:
            raise CaptureValidationError("score comparison tolerance must be positive")
        self.score_comparison_tolerance = score_comparison_tolerance

    def validate(self, capture_path: str | Path) -> ValidatedCapture:
        root = Path(capture_path)
        if not root.is_dir() or root.is_symlink():
            raise CaptureValidationError("capture generation must be a real directory")
        root_metadata = root.stat()
        if root_metadata.st_mode & 0o077:
            raise CaptureValidationError("capture generation is not access restricted")
        manifest_path = root / CAPTURE_MANIFEST_FILENAME
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise CaptureValidationError("capture manifest is missing")
        manifest = CaptureManifest.from_json_bytes(manifest_path.read_bytes())
        referenced_paths = tuple(manifest.artifact_refs.values())
        if len(referenced_paths) != len(set(referenced_paths)):
            raise CaptureValidationError("capture artifact refs are not one-to-one")
        if set(referenced_paths) != set(manifest.checksums):
            raise CaptureValidationError("capture checksum closure is not exact")
        expected_files = {CAPTURE_MANIFEST_FILENAME, *referenced_paths}
        actual_files: set[str] = set()
        for path in root.rglob("*"):
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                raise CaptureValidationError(f"capture contains a symlink: {relative}")
            if path.stat().st_mode & 0o077:
                raise CaptureValidationError(
                    f"capture path is not access restricted: {relative}"
                )
            if path.is_file():
                if not os.path.isfile(path):
                    raise CaptureValidationError(
                        f"capture contains a non-regular file: {relative}"
                    )
                if path.stat().st_nlink != 1:
                    raise CaptureValidationError(
                        f"capture contains a hard-linked file: {relative}"
                    )
                actual_files.add(relative)
            elif not path.is_dir():
                raise CaptureValidationError(
                    f"capture contains an unsupported file type: {relative}"
                )
        if actual_files != expected_files:
            raise CaptureValidationError(
                "capture file closure differs from its manifest: "
                f"missing={tuple(sorted(expected_files - actual_files))}, "
                f"unknown={tuple(sorted(actual_files - expected_files))}"
            )
        for relative, expected_digest in manifest.checksums.items():
            normalized = PurePosixPath(relative)
            if (
                normalized.is_absolute()
                or ".." in normalized.parts
                or normalized.as_posix() != relative
            ):
                raise CaptureValidationError("capture checksum path is unsafe")
            actual_digest = tree_or_blob_digest((root / relative).read_bytes())
            if actual_digest != expected_digest:
                raise CaptureValidationError(f"capture checksum mismatch: {relative}")

        descriptor = CaptureDescriptor.from_json_bytes(
            (root / manifest.artifact_refs["capture_descriptor"]).read_bytes()
        )
        self._validate_manifest_descriptor(manifest, descriptor)
        checkpoint = self._load_checkpoint(root, descriptor)
        history = ExperimentHistoryStore(
            str(root / descriptor.artifact_refs["experiment_history"])
        )
        events = self._load_events(
            root / descriptor.artifact_refs["execution_event_journal"]
        )
        archive_data = parse_json_bytes(
            (root / descriptor.artifact_refs["idea_archive"]).read_bytes()
        )
        if not isinstance(archive_data, dict):
            raise CaptureValidationError("idea archive must be an object")
        archive = IdeaArchiveState.from_dict(archive_data)
        raw_nodes = checkpoint.strategy_state.get("node_history")
        if not isinstance(raw_nodes, list):
            raise CaptureValidationError("checkpoint node history is unavailable")
        nodes = tuple(SearchNode.from_dict(item) for item in raw_nodes)
        if any(
            canonical_json_bytes(raw) != canonical_json_bytes(node.to_dict())
            for raw, node in zip(raw_nodes, nodes)
        ):
            raise CaptureValidationError("checkpoint node schema is not exact")
        branch_snapshots = tuple(
            BranchSnapshot.from_json_bytes((root / ref).read_bytes())
            for ref in descriptor.branch_snapshot_refs
        )
        self._validate_frontier(
            root,
            manifest,
            descriptor,
            checkpoint,
            history,
            events,
            archive,
            nodes,
            branch_snapshots,
        )
        return ValidatedCapture(
            path=root,
            directory_identity=(root_metadata.st_dev, root_metadata.st_ino),
            manifest=manifest,
            descriptor=descriptor,
            checkpoint=checkpoint,
            history=history,
            events=events,
            archive=archive,
            nodes=nodes,
            branch_snapshots=branch_snapshots,
        )

    @staticmethod
    def _validate_manifest_descriptor(
        manifest: CaptureManifest, descriptor: CaptureDescriptor
    ) -> None:
        fields = (
            "scope_contract_id",
            "scope_id",
            "run_id",
            "campaign_id",
            "capture_generation",
            "captured_at",
        )
        conflicts = tuple(
            name
            for name in fields
            if getattr(manifest, name) != getattr(descriptor, name)
        )
        if conflicts:
            raise CaptureValidationError(
                f"capture manifest and descriptor conflict: {conflicts}"
            )
        if dict(manifest.artifact_refs) != dict(descriptor.artifact_refs):
            raise CaptureValidationError("capture artifact refs changed in descriptor")
        for name in (
            "checkpoint",
            "execution_event_journal",
            "idea_archive",
            "experiment_history",
        ):
            if (
                descriptor.artifact_completeness.get(name)
                is not ArtifactCompleteness.PRESENT
            ):
                raise CaptureValidationError(
                    f"capture core authority is not present: {name}"
                )

    @staticmethod
    def _load_checkpoint(root: Path, descriptor: CaptureDescriptor) -> RunCheckpoint:
        parsed = parse_json_bytes(
            (root / descriptor.artifact_refs["checkpoint"]).read_bytes()
        )
        if not isinstance(parsed, dict):
            raise CaptureValidationError("checkpoint must be an object")
        checkpoint = RunCheckpoint.from_dict(parsed)
        if canonical_json_bytes(parsed) != canonical_json_bytes(checkpoint.to_dict()):
            raise CaptureValidationError("checkpoint schema is not exact")
        if (
            checkpoint.strategy_type != "generic"
            or set(checkpoint.strategy_state) != GENERIC_SEARCH_STATE_FIELDS
            or checkpoint.strategy_state.get("schema") != GENERIC_SEARCH_STATE_SCHEMA
        ):
            raise CaptureValidationError("checkpoint strategy schema is not exact")
        return checkpoint

    @staticmethod
    def _load_events(path: Path) -> tuple[ExecutionRevisionEvent, ...]:
        payload = path.read_bytes()
        if payload and not payload.endswith(b"\n"):
            raise CaptureValidationError("execution journal has an incomplete tail")
        lines = payload.splitlines()
        if any(not line for line in lines):
            raise CaptureValidationError("execution journal contains a blank event")
        events = tuple(ExecutionRevisionEvent.from_json_bytes(line) for line in lines)
        expected: dict[int, int] = {}
        first_nodes: list[int] = []
        for event in events:
            raw_projection = to_json_value(event.projection)
            record = ExperimentRecord.from_dict(raw_projection)
            if canonical_json_bytes(raw_projection) != canonical_json_bytes(
                record.to_dict()
            ):
                raise CaptureValidationError(
                    "execution journal projection schema is not exact"
                )
            revision = expected.get(event.node_id, 0)
            if event.execution_revision != revision:
                raise CaptureValidationError(
                    "execution journal revisions are not gap-free"
                )
            if event.node_id not in expected:
                first_nodes.append(event.node_id)
            expected[event.node_id] = revision + 1
        if first_nodes != list(range(len(first_nodes))):
            raise CaptureValidationError(
                "execution journal node ids are not contiguous"
            )
        return events

    def _validate_frontier(
        self,
        root: Path,
        manifest: CaptureManifest,
        descriptor: CaptureDescriptor,
        checkpoint: RunCheckpoint,
        history: ExperimentHistoryStore,
        events: tuple[ExecutionRevisionEvent, ...],
        archive: IdeaArchiveState,
        nodes: tuple[SearchNode, ...],
        branches: tuple[BranchSnapshot, ...],
    ) -> None:
        if (
            history.run_id != manifest.run_id
            or history.campaign_id != manifest.campaign_id
            or archive.campaign_id != manifest.campaign_id
        ):
            raise CaptureValidationError("capture authority identity mismatch")
        if any(
            event.run_id != manifest.run_id or event.campaign_id != manifest.campaign_id
            for event in events
        ):
            raise CaptureValidationError("execution journal identity mismatch")
        if descriptor.completion_state is CompletionState.COMPLETE:
            if checkpoint.status != "completed":
                raise CaptureValidationError(
                    "complete capture has a resumable checkpoint"
                )
        elif checkpoint.status != "running":
            raise CaptureValidationError(
                "stopped/crashed capture has a completed checkpoint"
            )
        if manifest.checkpoint_frontier != checkpoint.completed_iterations:
            raise CaptureValidationError("checkpoint frontier watermark is false")
        strategy_iteration_count = checkpoint.strategy_state.get("iteration_count")
        expected_watermarks = {
            "branch_snapshot_count": len(branches),
            "checkpoint_completed_iterations": checkpoint.completed_iterations,
            "checkpoint_node_count": len(nodes),
            "execution_journal_event_count": len(events),
            "experiment_history_count": len(history.experiments),
            "experiment_history_revision": history.revision,
            "idea_archive_revision": archive.revision,
            "strategy_iteration_count": strategy_iteration_count,
        }
        if dict(manifest.capture_watermarks) != expected_watermarks:
            raise CaptureValidationError("capture watermarks do not match authorities")
        records = tuple(history.experiments)
        revision_records = tuple(
            ExperimentRecord.from_dict(to_json_value(event.projection))
            for event in events
        )
        referenced_evaluation_fingerprints = validate_evaluation_fingerprints(
            revision_records,
            descriptor.evaluation_fingerprints,
            self.score_comparison_tolerance,
            CaptureValidationError,
        )
        if referenced_evaluation_fingerprints != descriptor.evaluation_fingerprints:
            raise CaptureValidationError(
                "evaluation fingerprint registry is not an exact referenced closure"
            )
        if len(nodes) != len(records):
            raise CaptureValidationError("checkpoint/history node counts differ")
        validate_execution_provenance(
            archive,
            records,
            events,
            nodes,
            CaptureValidationError,
        )
        terminals: dict[int, ExecutionRevisionEvent] = {}
        for event in events:
            terminals[event.node_id] = event
        if tuple(sorted(terminals)) != tuple(range(len(records))):
            raise CaptureValidationError("journal terminal node frontier is incomplete")
        ideas = {idea.idea_id: idea for idea in archive.ideas}
        node_links: dict[int, str] = {}
        for idea in archive.ideas:
            if idea.experiment_node_id is not None:
                if idea.experiment_node_id in node_links:
                    raise CaptureValidationError("multiple ideas link to one node")
                node_links[idea.experiment_node_id] = idea.idea_id
        branch_by_revision = {
            (branch.node_id, branch.execution_revision): branch for branch in branches
        }
        if len(branch_by_revision) != len(branches):
            raise CaptureValidationError("capture contains duplicate branch snapshots")
        expected_completeness = {
            "checkpoint",
            "execution_event_journal",
            "idea_archive",
            "experiment_history",
            *(f"branch:{event.node_id}:{event.execution_revision}" for event in events),
            *(f"run_log:{path}" for path in descriptor.run_log_refs),
        }
        if set(descriptor.artifact_completeness) != expected_completeness:
            raise CaptureValidationError("artifact completeness closure is not exact")
        source_payload_refs: set[str] = set()
        for node, record in zip(nodes, records):
            projected = ExperimentRecord.from_node(
                node,
                record.objective_direction,
                history.require_idea_links,
                record.solution_embedding,
            )
            if projected != record:
                raise CaptureValidationError("checkpoint node projection changed")
            terminal = terminals[record.node_id]
            if canonical_json_bytes(terminal.projection) != canonical_json_bytes(
                record.to_dict()
            ):
                raise CaptureValidationError(
                    "journal terminal projection differs from experiment history"
                )
            if record.idea_id is None or record.idea_id not in ideas:
                raise CaptureValidationError("executed node has no selected idea")
            idea = ideas[record.idea_id]
            if (
                node_links.get(record.node_id) != record.idea_id
                or idea.selected_in_batch_id != record.selection_batch_id
                or idea.experiment_node_id != record.node_id
            ):
                raise CaptureValidationError("idea/node linkage is not one-to-one")
            expected_evaluators = tuple(
                sorted({attempt.evaluator_id for attempt in record.evaluation_attempts})
            )
            if terminal.evaluator_fingerprint_ids != expected_evaluators:
                raise CaptureValidationError("journal evaluator fingerprints changed")
            if record.evaluation_valid and record.raw_score is not None:
                if not record.evaluation_attempts:
                    raise CaptureValidationError(
                        "valid measured experiment has no evaluator fingerprint"
                    )
                if terminal.measurements.get("raw_score") != record.raw_score:
                    raise CaptureValidationError("journal raw measurement changed")
            expected_measurements = dict(record.metrics)
            if record.raw_score is not None:
                expected_measurements["raw_score"] = record.raw_score
            if dict(terminal.measurements) != expected_measurements:
                raise CaptureValidationError("journal measurements changed")
            expected_terminal_refs = {
                name: value
                for name, value in {
                    "branch": node.branch_name,
                    "parent_branch": node.parent_branch_name,
                    "implementation_base": node.implementation_base_ref,
                    "diff_base": node.diff_base_ref,
                    "feedback_base": node.feedback_base_ref,
                }.items()
                if value
            }
            for position, attempt in enumerate(record.evaluation_attempts):
                expected_terminal_refs[f"evaluation_commit_{position}"] = (
                    attempt.commit_sha
                )
            if any(
                terminal.artifact_refs.get(name) != value
                for name, value in expected_terminal_refs.items()
            ):
                raise CaptureValidationError(
                    "journal terminal artifact provenance changed"
                )
        source_payload_refs: set[str] = set()
        for event in events:
            revision_record = ExperimentRecord.from_dict(
                to_json_value(event.projection)
            )
            branch_key = (
                f"branch:{revision_record.node_id}:"
                f"{revision_record.execution_revision}"
            )
            completeness = descriptor.artifact_completeness.get(branch_key)
            branch = branch_by_revision.get(
                (revision_record.node_id, revision_record.execution_revision)
            )
            if branch is None:
                if (
                    completeness is not ArtifactCompleteness.UNAVAILABLE
                    or not revision_record.had_error
                ):
                    raise CaptureValidationError("branch completeness is false")
                continue
            if completeness is not ArtifactCompleteness.PRESENT:
                raise CaptureValidationError("present branch is not declared present")
            if (
                descriptor.artifact_refs.get(branch_key)
                not in descriptor.branch_snapshot_refs
            ):
                raise CaptureValidationError(
                    "branch logical ref does not name its manifest"
                )
            source_payload_refs.update(
                validate_branch_evidence(
                    read_ref=lambda relative_path: (root / relative_path).read_bytes(),
                    descriptor=descriptor,
                    record=revision_record,
                    event=event,
                    branch=branch,
                    error_type=CaptureValidationError,
                )
            )
        if set(node_links) != set(range(len(records))):
            raise CaptureValidationError(
                "archive node linkage exceeds captured frontier"
            )
        structural_refs = {
            descriptor.artifact_refs["capture_descriptor"],
            descriptor.artifact_refs["checkpoint"],
            descriptor.artifact_refs["execution_event_journal"],
            descriptor.artifact_refs["idea_archive"],
            descriptor.artifact_refs["experiment_history"],
            *descriptor.branch_snapshot_refs,
            *descriptor.run_log_refs,
        }
        unexplained_refs = set(descriptor.artifact_refs.values()) - structural_refs
        if unexplained_refs != source_payload_refs:
            raise CaptureValidationError("source payload closure is not exact")
