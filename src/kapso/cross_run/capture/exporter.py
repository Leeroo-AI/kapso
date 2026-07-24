"""Reconcile one durable run frontier and atomically capture it in quarantine."""

from __future__ import annotations

import fcntl
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    normalize_utc_timestamp,
    parse_json_bytes,
    require_content_id,
    require_identifier,
    source_tree_digest,
    tree_or_blob_digest,
    to_json_value,
)
from kapso.cross_run.capture.journal import ExecutionRevisionJournal
from kapso.cross_run.capture.git_evidence import parse_commit_object
from kapso.cross_run.capture.provenance import validate_execution_provenance
from kapso.cross_run.capture.evaluation_evidence import (
    validate_evaluation_fingerprints,
)
from kapso.cross_run.capture.safety import (
    path_matches_denied_pattern,
    read_restricted_regular_file,
)
from kapso.cross_run.contracts import (
    ArtifactCompleteness,
    ArtifactEnvironment,
    CaptureManifest,
    CompletionState,
    EvaluationFingerprint,
    StrictContract,
    TaskContextBinding,
)
from kapso.cross_run.git_refs import git_object_sha, require_git_ref_name
from kapso.cross_run.git_command import BoundedGitCommand
from kapso.cross_run.github.command import CommandOutputKind, CommandRunner
from kapso.cross_run.settings import CaptureSettings, SanitationSettings
from kapso.execution.memories.experiment_memory.record import (
    EXPERIMENT_HISTORY_SCHEMA,
    ExperimentRecord,
)
from kapso.execution.memories.experiment_memory.store import ExperimentHistoryStore
from kapso.execution.run_checkpoint import RunCheckpoint
from kapso.execution.search_strategies.node import SearchNode
from kapso.execution.search_strategies.generic.ideation.archive import (
    IdeaArchiveState,
)

CAPTURE_DESCRIPTOR_SCHEMA = "kapso.capture_descriptor.v1"
BRANCH_SNAPSHOT_SCHEMA = "kapso.branch_snapshot.v1"
CAPTURE_MANIFEST_FILENAME = "capture_manifest.json"
CAPTURE_DESCRIPTOR_REF = "payload/capture_descriptor.json"
CAPTURE_CURRENT_FILENAME = "current.json"
CAPTURE_EXPORT_LOCK_FILENAME = ".export.lock"
_CORE_ARTIFACT_REFS = {
    "checkpoint": "payload/checkpoint.json",
    "execution_event_journal": "payload/execution_events.jsonl",
    "idea_archive": "payload/idea_archive.json",
    "experiment_history": "payload/experiment_history.json",
}


class CaptureExportError(ValueError):
    """The requested frontier cannot be captured without ambiguity."""


def _utc_now() -> str:
    current = datetime.now(timezone.utc)
    timespec = "microseconds" if current.microsecond else "seconds"
    return current.isoformat(timespec=timespec).replace("+00:00", "Z")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    os.fsync(descriptor)
    os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        directory.chmod(0o700)
        _fsync_directory(directory)
    root.chmod(0o700)
    _fsync_directory(root)


def _storage_key(identifier: str) -> str:
    return tree_or_blob_digest(identifier.encode("utf-8"))[7:]


def _reject_symlink_components(path: Path) -> None:
    for component in (path, *path.parents):
        if component.is_symlink():
            raise CaptureExportError("capture storage path contains a symlink")


def _write_bytes(path: Path, payload: bytes, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(mode)


def _write_atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(canonical_json_bytes(payload))
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.chmod(0o600)
    os.replace(temporary, path)
    path.chmod(0o600)
    _fsync_directory(path.parent)


@dataclass(frozen=True)
class CaptureDescriptor(StrictContract):
    """Bundle metadata that is outside the frozen CaptureManifest contract."""

    schema: str
    scope_contract_id: str
    scope_id: str
    run_id: str
    campaign_id: str
    completion_state: CompletionState
    capture_generation: int
    started_at: str
    captured_at: str
    kapso_commit: str
    launch_manifest_id: str
    knowledge_snapshot_id: str
    expert_base_release_id: str
    task_context_binding: TaskContextBinding
    artifact_environment: ArtifactEnvironment
    evaluation_fingerprints: tuple[EvaluationFingerprint, ...]
    artifact_completeness: Mapping[str, ArtifactCompleteness]
    artifact_refs: Mapping[str, str]
    branch_snapshot_refs: tuple[str, ...]
    run_log_refs: tuple[str, ...]

    def _validate(self) -> None:
        if self.schema != CAPTURE_DESCRIPTOR_SCHEMA:
            raise CaptureExportError("capture descriptor schema is incompatible")
        require_content_id(self.scope_contract_id, "scope_contract_id")
        for name in ("scope_id", "run_id", "campaign_id"):
            require_identifier(getattr(self, name), name)
        if type(self.capture_generation) is not int or self.capture_generation < 0:
            raise CaptureExportError("capture generation must be non-negative")
        normalize_utc_timestamp(self.started_at, "started_at")
        normalize_utc_timestamp(self.captured_at, "captured_at")
        if not self.kapso_commit:
            raise CaptureExportError("kapso commit must not be empty")
        for name in (
            "launch_manifest_id",
            "knowledge_snapshot_id",
            "expert_base_release_id",
        ):
            require_content_id(getattr(self, name), name)
        if self.task_context_binding.scope_contract_id != self.scope_contract_id:
            raise CaptureExportError("capture context uses another scope contract")
        if self.task_context_binding.scope_id != self.scope_id:
            raise CaptureExportError("capture context uses another scope")
        if (
            self.artifact_environment.expert_base_release_id
            != self.expert_base_release_id
        ):
            raise CaptureExportError("capture environment uses another expert release")
        if self.artifact_environment.kapso_commit != self.kapso_commit:
            raise CaptureExportError("capture environment uses another Kapso commit")
        fingerprint_ids = tuple(
            item.evaluation_fingerprint_id for item in self.evaluation_fingerprints
        )
        if fingerprint_ids != tuple(sorted(set(fingerprint_ids))):
            raise CaptureExportError(
                "capture evaluator fingerprints must be sorted and unique"
            )
        if not self.artifact_completeness or not self.artifact_refs:
            raise CaptureExportError("capture descriptor artifacts must not be empty")
        referenced = set(self.branch_snapshot_refs) | set(self.run_log_refs)
        if not referenced.issubset(self.artifact_refs.values()):
            raise CaptureExportError(
                "capture descriptor references an unknown artifact"
            )


@dataclass(frozen=True)
class BranchSnapshot(StrictContract):
    """Exact Git commit and allowlisted regular-file closure for one revision."""

    schema: str
    node_id: int
    execution_revision: int
    branch_name: str
    parent_branch_name: str
    revision_ref: str
    commit_sha: str
    implementation_base_ref: str
    diff_base_ref: str
    feedback_base_ref: str
    base_commit_shas: Mapping[str, str]
    evaluated_commit_shas: tuple[str, ...]
    root_tree_sha: str
    commit_objects: tuple[Mapping[str, Any], ...]
    source_tree_digest: str
    source_files: tuple[Mapping[str, Any], ...]
    excluded_files: tuple[Mapping[str, Any], ...]

    def _validate(self) -> None:
        if self.schema != BRANCH_SNAPSHOT_SCHEMA:
            raise CaptureExportError("branch snapshot schema is incompatible")
        for value, name in (
            (self.node_id, "node_id"),
            (self.execution_revision, "execution_revision"),
        ):
            if type(value) is not int or value < 0:
                raise CaptureExportError(f"branch snapshot {name} is invalid")
        require_git_ref_name(
            self.branch_name,
            "branch_name",
            qualified=False,
            error_type=CaptureExportError,
        )
        if self.parent_branch_name:
            require_git_ref_name(
                self.parent_branch_name,
                "parent_branch_name",
                qualified=False,
                error_type=CaptureExportError,
            )
        require_git_ref_name(
            self.revision_ref,
            "revision_ref",
            qualified=True,
            error_type=CaptureExportError,
        )
        if not re.fullmatch(r"[0-9a-f]{40}", self.commit_sha):
            raise CaptureExportError("branch snapshot commit is invalid")
        expected_base_keys = {
            name
            for name, value in {
                "implementation": self.implementation_base_ref,
                "diff": self.diff_base_ref,
                "feedback": self.feedback_base_ref,
            }.items()
            if value
        }
        if set(self.base_commit_shas) != expected_base_keys or any(
            re.fullmatch(r"[0-9a-f]{40}", value) is None
            for value in self.base_commit_shas.values()
        ):
            raise CaptureExportError("branch snapshot base commit closure is invalid")
        if self.evaluated_commit_shas != tuple(sorted(set(self.evaluated_commit_shas))):
            raise CaptureExportError("evaluated commits must be sorted and unique")
        if any(
            not re.fullmatch(r"[0-9a-f]{40}", value)
            for value in self.evaluated_commit_shas
        ):
            raise CaptureExportError("evaluated commit is invalid")
        if any(value != self.commit_sha for value in self.evaluated_commit_shas):
            raise CaptureExportError("evaluation and captured branch commits differ")
        if re.fullmatch(r"[0-9a-f]{40}", self.root_tree_sha) is None:
            raise CaptureExportError("branch root tree id is invalid")
        commit_fields = {"commit_sha", "payload_ref"}
        if not self.commit_objects or any(
            set(item) != commit_fields for item in self.commit_objects
        ):
            raise CaptureExportError("branch commit-object evidence is invalid")
        commit_shas = tuple(item["commit_sha"] for item in self.commit_objects)
        commit_refs = tuple(item["payload_ref"] for item in self.commit_objects)
        if (
            self.commit_sha not in commit_shas
            or len(set(commit_shas)) != len(commit_shas)
            or len(set(commit_refs)) != len(commit_refs)
            or any(
                re.fullmatch(r"[0-9a-f]{40}", value) is None for value in commit_shas
            )
        ):
            raise CaptureExportError("branch commit-object closure is invalid")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.source_tree_digest):
            raise CaptureExportError("branch source tree digest is invalid")
        source_fields = {
            "git_blob_sha",
            "mode",
            "payload_ref",
            "sha256",
            "size",
            "source_path",
        }
        exclusion_fields = {
            "git_object_sha",
            "mode",
            "object_type",
            "path",
            "reason",
            "size",
        }
        if any(set(item) != source_fields for item in self.source_files):
            raise CaptureExportError("branch source file descriptor is invalid")
        if any(set(item) != exclusion_fields for item in self.excluded_files):
            raise CaptureExportError("branch exclusion descriptor is invalid")
        source_paths = tuple(item["source_path"] for item in self.source_files)
        excluded_paths = tuple(item["path"] for item in self.excluded_files)
        if len(set(source_paths)) != len(source_paths) or len(
            set(excluded_paths)
        ) != len(excluded_paths):
            raise CaptureExportError("branch snapshot paths must be unique")
        if set(source_paths) & set(excluded_paths):
            raise CaptureExportError("branch path is both captured and excluded")
        for path in source_paths + excluded_paths:
            if not isinstance(path, str):
                raise CaptureExportError("branch snapshot path is unsafe")
            normalized = PurePosixPath(path)
            if (
                normalized.is_absolute()
                or ".." in normalized.parts
                or normalized.as_posix() != path
            ):
                raise CaptureExportError("branch snapshot path is unsafe")


@dataclass(frozen=True)
class RunCaptureRequest:
    workspace_dir: Path
    idea_archive_path: Path
    scope_contract_id: str
    scope_id: str
    run_id: str
    campaign_id: str
    configuration_fingerprint: str
    completion_state: CompletionState
    started_at: str
    kapso_commit: str
    launch_manifest_id: str
    knowledge_snapshot_id: str
    expert_base_release_id: str
    task_context_binding: TaskContextBinding
    artifact_environment: ArtifactEnvironment
    evaluation_fingerprints: tuple[EvaluationFingerprint, ...]
    run_log_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "workspace_dir", Path(self.workspace_dir))
        object.__setattr__(self, "idea_archive_path", Path(self.idea_archive_path))
        require_content_id(self.scope_contract_id, "scope_contract_id")
        for name in ("scope_id", "run_id", "campaign_id"):
            require_identifier(getattr(self, name), name)
        if not isinstance(self.completion_state, CompletionState):
            raise CaptureExportError("completion_state must use the strict enum")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.configuration_fingerprint):
            raise CaptureExportError("configuration fingerprint is invalid")
        normalize_utc_timestamp(self.started_at, "started_at")
        if not self.kapso_commit:
            raise CaptureExportError("kapso commit must not be empty")
        for name in (
            "launch_manifest_id",
            "knowledge_snapshot_id",
            "expert_base_release_id",
        ):
            require_content_id(getattr(self, name), name)
        if (
            self.task_context_binding.scope_contract_id != self.scope_contract_id
            or self.task_context_binding.scope_id != self.scope_id
        ):
            raise CaptureExportError("capture request context uses another scope")
        if (
            self.artifact_environment.expert_base_release_id
            != self.expert_base_release_id
        ):
            raise CaptureExportError("capture request environment uses another release")
        if self.artifact_environment.kapso_commit != self.kapso_commit:
            raise CaptureExportError(
                "capture request environment uses another Kapso commit"
            )
        fingerprint_ids = tuple(
            item.evaluation_fingerprint_id for item in self.evaluation_fingerprints
        )
        if fingerprint_ids != tuple(sorted(set(fingerprint_ids))):
            raise CaptureExportError(
                "capture evaluator fingerprints must be sorted and unique"
            )
        if len(set(self.run_log_paths)) != len(self.run_log_paths):
            raise CaptureExportError("run log paths must be unique")
        for path in self.run_log_paths:
            normalized = PurePosixPath(path)
            if (
                normalized.is_absolute()
                or ".." in normalized.parts
                or normalized.as_posix() != path
            ):
                raise CaptureExportError("run log path must be relative to workspace")


@dataclass(frozen=True)
class ExportedCapture:
    path: Path
    manifest: CaptureManifest
    descriptor: CaptureDescriptor


class RunCaptureExporter:
    """Publish only a mutually reconciled authority frontier."""

    def __init__(
        self,
        capture_settings: CaptureSettings,
        sanitation_settings: SanitationSettings,
        git_command_runner: CommandRunner | None = None,
    ):
        self.capture_settings = capture_settings
        self.sanitation_settings = sanitation_settings
        self.git_command = BoundedGitCommand(
            timeout_seconds=capture_settings.git_command_timeout_seconds,
            maximum_output_bytes=capture_settings.git_command_output_bytes,
            runner=git_command_runner,
        )

    def export(self, request: RunCaptureRequest) -> ExportedCapture:
        quarantine_root = request.workspace_dir / self.capture_settings.quarantine_path
        _reject_symlink_components(quarantine_root)
        quarantine_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        quarantine_root.chmod(0o700)
        runs_root = quarantine_root / "runs"
        runs_root.mkdir(exist_ok=True, mode=0o700)
        runs_root.chmod(0o700)
        run_root = runs_root / _storage_key(request.run_id)
        run_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        run_root.chmod(0o700)
        lock_path = run_root / CAPTURE_EXPORT_LOCK_FILENAME
        _reject_symlink_components(lock_path)
        with lock_path.open("a+", encoding="utf-8") as lock_handle:
            lock_path.chmod(0o600)
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            exported = self._export_locked(request, run_root)
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        return exported

    def _export_locked(
        self, request: RunCaptureRequest, run_root: Path
    ) -> ExportedCapture:
        authorities = self._load_reconciled_authorities(request)
        current = self._load_current(run_root, request)
        self._remove_uncommitted_generations(run_root, current)
        generation = 0 if current is None else current.manifest.capture_generation + 1
        staging = Path(tempfile.mkdtemp(prefix=".capture.", dir=run_root))
        staging.chmod(0o700)
        artifact_refs: dict[str, str] = dict(_CORE_ARTIFACT_REFS)
        checksums: dict[str, str] = {}
        authority_payloads = {
            _CORE_ARTIFACT_REFS["checkpoint"]: authorities["checkpoint_payload"],
            _CORE_ARTIFACT_REFS["execution_event_journal"]: authorities[
                "journal_payload"
            ],
            _CORE_ARTIFACT_REFS["idea_archive"]: authorities["archive_payload"],
            _CORE_ARTIFACT_REFS["experiment_history"]: authorities["history_payload"],
        }
        for relative_path, payload in authority_payloads.items():
            _write_bytes(staging / relative_path, payload)
            checksums[relative_path] = tree_or_blob_digest(payload)

        branch_refs, branch_completeness = self._capture_branches(
            staging,
            request,
            authorities["events"],
            artifact_refs,
            checksums,
        )
        run_log_refs = self._capture_run_logs(
            staging,
            request,
            artifact_refs,
            checksums,
        )
        completeness: dict[str, ArtifactCompleteness] = {
            name: ArtifactCompleteness.PRESENT for name in _CORE_ARTIFACT_REFS
        }
        completeness.update(branch_completeness)
        completeness.update(
            {f"run_log:{path}": ArtifactCompleteness.PRESENT for path in run_log_refs}
        )
        captured_at = _utc_now()
        artifact_refs["capture_descriptor"] = CAPTURE_DESCRIPTOR_REF
        descriptor = CaptureDescriptor(
            schema=CAPTURE_DESCRIPTOR_SCHEMA,
            scope_contract_id=request.scope_contract_id,
            scope_id=request.scope_id,
            run_id=request.run_id,
            campaign_id=request.campaign_id,
            completion_state=request.completion_state,
            capture_generation=generation,
            started_at=request.started_at,
            captured_at=captured_at,
            kapso_commit=request.kapso_commit,
            launch_manifest_id=request.launch_manifest_id,
            knowledge_snapshot_id=request.knowledge_snapshot_id,
            expert_base_release_id=request.expert_base_release_id,
            task_context_binding=request.task_context_binding,
            artifact_environment=request.artifact_environment,
            evaluation_fingerprints=authorities["evaluation_fingerprints"],
            artifact_completeness=completeness,
            artifact_refs=artifact_refs,
            branch_snapshot_refs=tuple(branch_refs),
            run_log_refs=tuple(run_log_refs),
        )
        descriptor_payload = descriptor.to_json_bytes()
        _write_bytes(staging / CAPTURE_DESCRIPTOR_REF, descriptor_payload)
        checksums[CAPTURE_DESCRIPTOR_REF] = tree_or_blob_digest(descriptor_payload)
        watermarks = {
            "branch_snapshot_count": len(branch_refs),
            "checkpoint_completed_iterations": authorities[
                "checkpoint"
            ].completed_iterations,
            "checkpoint_node_count": len(authorities["nodes"]),
            "execution_journal_event_count": len(authorities["events"]),
            "experiment_history_count": len(authorities["records"]),
            "experiment_history_revision": authorities["history_revision"],
            "idea_archive_revision": authorities["archive"].revision,
            "strategy_iteration_count": authorities["strategy_iteration_count"],
        }
        manifest = CaptureManifest.mint(
            scope_contract_id=request.scope_contract_id,
            scope_id=request.scope_id,
            run_id=request.run_id,
            campaign_id=request.campaign_id,
            capture_generation=generation,
            supersedes_capture_manifest_id=(
                current.manifest.capture_manifest_id if current is not None else None
            ),
            checkpoint_frontier=authorities["checkpoint"].completed_iterations,
            capture_watermarks=watermarks,
            configuration_fingerprint=request.configuration_fingerprint,
            artifact_refs=artifact_refs,
            checksums=checksums,
            captured_at=captured_at,
        )
        _write_bytes(
            staging / CAPTURE_MANIFEST_FILENAME,
            manifest.to_json_bytes(),
        )
        total_size = sum(
            path.stat().st_size for path in staging.rglob("*") if path.is_file()
        )
        if total_size > self.capture_settings.bundle_asset_size_bytes:
            raise CaptureExportError("capture generation byte limit exceeded")
        self._require_authorities_unchanged(request, authorities)
        for event in authorities["events"]:
            self._resolve_pinned_commit(request.workspace_dir, event)

        if current is not None and self._same_frontier(current, manifest, descriptor):
            shutil.rmtree(staging)
            return current

        destination = run_root / f"generation-{generation:020d}"
        if destination.exists():
            raise CaptureExportError("capture generation already exists")
        _fsync_tree(staging)
        os.replace(staging, destination)
        _fsync_directory(run_root)
        marker = {
            "capture_manifest_id": manifest.capture_manifest_id,
            "generation": generation,
            "path": destination.name,
        }
        _write_atomic_json(run_root / CAPTURE_CURRENT_FILENAME, marker)
        return ExportedCapture(
            path=destination, manifest=manifest, descriptor=descriptor
        )

    def _load_reconciled_authorities(
        self, request: RunCaptureRequest
    ) -> dict[str, Any]:
        checkpoint_path = request.workspace_dir / self.capture_settings.checkpoint_path
        history_path = (
            request.workspace_dir / self.capture_settings.experiment_history_path
        )
        journal_path = (
            request.workspace_dir
            / self.capture_settings.state_path
            / self.capture_settings.journal_filename
        )
        required = (
            checkpoint_path,
            history_path,
            journal_path,
            request.idea_archive_path,
        )
        workspace = request.workspace_dir.resolve()
        unsafe = tuple(
            str(path)
            for path in required
            if path.is_symlink()
            or (workspace != path.resolve() and workspace not in path.resolve().parents)
        )
        if unsafe:
            raise CaptureExportError(f"capture authority paths are unsafe: {unsafe}")
        missing = tuple(str(path) for path in required if not path.is_file())
        if missing:
            raise CaptureExportError(f"capture authorities are missing: {missing}")
        source_payloads = {
            "checkpoint_source_payload": checkpoint_path.read_bytes(),
            "history_source_payload": history_path.read_bytes(),
            "journal_source_payload": journal_path.read_bytes(),
            "archive_source_payload": request.idea_archive_path.read_bytes(),
        }
        checkpoint_data = parse_json_bytes(source_payloads["checkpoint_source_payload"])
        if not isinstance(checkpoint_data, dict):
            raise CaptureExportError("checkpoint must be an object")
        checkpoint = RunCheckpoint.from_dict(checkpoint_data)
        history = ExperimentHistoryStore(str(history_path))
        if (
            history.run_id != request.run_id
            or history.campaign_id != request.campaign_id
        ):
            raise CaptureExportError(
                "experiment history identity conflicts with request"
            )
        journal = ExecutionRevisionJournal(
            journal_path,
            run_id=request.run_id,
            campaign_id=request.campaign_id,
        )
        events = journal.read_events()
        if len(events) not in {history.revision, history.revision + 1}:
            raise CaptureExportError(
                "journal and experiment history are not at a recoverable boundary"
            )
        durable_events = events[: history.revision]
        durable_records = self._terminal_records(durable_events)
        if tuple(history.experiments) != durable_records:
            raise CaptureExportError("journal and experiment history do not reconcile")
        archive_data = parse_json_bytes(source_payloads["archive_source_payload"])
        if not isinstance(archive_data, dict):
            raise CaptureExportError("idea archive must be an object")
        live_archive = IdeaArchiveState.from_dict(archive_data)
        if live_archive.campaign_id != request.campaign_id:
            raise CaptureExportError("idea archive campaign conflicts with request")
        strategy_state = checkpoint.strategy_state
        snapshot_data = strategy_state.get("idea_archive_snapshot")
        if not isinstance(snapshot_data, dict):
            raise CaptureExportError("checkpoint has no exact idea archive snapshot")
        archive = IdeaArchiveState.from_dict(snapshot_data)
        if archive.campaign_id != request.campaign_id:
            raise CaptureExportError(
                "checkpoint archive campaign conflicts with request"
            )
        if live_archive.revision < archive.revision:
            raise CaptureExportError("live idea archive is behind the checkpoint")
        if live_archive.revision == archive.revision and live_archive != archive:
            raise CaptureExportError("live idea archive conflicts with the checkpoint")
        raw_nodes = strategy_state.get("node_history")
        if not isinstance(raw_nodes, list):
            raise CaptureExportError("checkpoint has no node history frontier")
        nodes = tuple(SearchNode.from_dict(node) for node in raw_nodes)
        prefix_length = sum(node.execution_revision + 1 for node in nodes)
        if prefix_length > history.revision:
            raise CaptureExportError("checkpoint exceeds durable experiment history")
        selected_events = events[:prefix_length]
        records = self._terminal_records(selected_events)
        if len(nodes) != len(records):
            raise CaptureExportError("checkpoint and journal-prefix counts differ")
        for node, record in zip(nodes, records):
            projected = ExperimentRecord.from_node(
                node,
                record.objective_direction,
                history.require_idea_links,
                record.solution_embedding,
            )
            if projected != record:
                raise CaptureExportError(
                    "checkpoint node conflicts with the global journal prefix"
                )
        if request.completion_state is CompletionState.COMPLETE and (
            history.revision != prefix_length or len(events) != prefix_length
        ):
            raise CaptureExportError(
                "complete capture has uncheckpointed authority tails"
            )
        ideas = {idea.idea_id: idea for idea in archive.ideas}
        for record in records:
            if record.idea_id is None or record.idea_id not in ideas:
                raise CaptureExportError(
                    "experiment history references an unknown idea"
                )
            idea = ideas[record.idea_id]
            if (
                idea.experiment_node_id != record.node_id
                or idea.selected_in_batch_id != record.selection_batch_id
            ):
                raise CaptureExportError("idea archive and experiment history diverged")
        revision_records = tuple(
            ExperimentRecord.from_dict(to_json_value(event.projection))
            for event in selected_events
        )
        evaluation_fingerprints = validate_evaluation_fingerprints(
            revision_records,
            request.evaluation_fingerprints,
            self.capture_settings.score_comparison_tolerance,
            CaptureExportError,
        )
        validate_execution_provenance(
            archive,
            records,
            selected_events,
            nodes,
            CaptureExportError,
        )
        iteration_count = strategy_state.get("iteration_count")
        if type(iteration_count) is not int or iteration_count < len(nodes):
            raise CaptureExportError(
                "checkpoint strategy iteration watermark is invalid"
            )
        history_payload = canonical_json_bytes(
            {
                "schema": EXPERIMENT_HISTORY_SCHEMA,
                "run_id": history.run_id,
                "campaign_id": history.campaign_id,
                "revision": prefix_length,
                "objective_direction": history.objective_direction,
                "require_idea_links": history.require_idea_links,
                "records": [record.to_dict() for record in records],
            }
        )
        journal_payload = b"".join(
            canonical_json_bytes(event.to_dict()) + b"\n" for event in selected_events
        )
        return {
            **source_payloads,
            "checkpoint_payload": source_payloads["checkpoint_source_payload"],
            "history_payload": history_payload,
            "journal_payload": journal_payload,
            "archive_payload": canonical_json_bytes(archive.to_dict()),
            "checkpoint": checkpoint,
            "history": history,
            "history_revision": prefix_length,
            "events": selected_events,
            "records": records,
            "archive": archive,
            "nodes": nodes,
            "strategy_iteration_count": iteration_count,
            "evaluation_fingerprints": evaluation_fingerprints,
            "authority_paths": {
                "checkpoint_source_payload": checkpoint_path,
                "history_source_payload": history_path,
                "journal_source_payload": journal_path,
                "archive_source_payload": request.idea_archive_path,
            },
        }

    @staticmethod
    def _terminal_records(events: tuple[Any, ...]) -> tuple[ExperimentRecord, ...]:
        terminal: dict[int, ExperimentRecord] = {}
        for event in events:
            terminal[event.node_id] = ExperimentRecord.from_dict(
                to_json_value(event.projection)
            )
        if tuple(sorted(terminal)) != tuple(range(len(terminal))):
            raise CaptureExportError("journal prefix node ids are not contiguous")
        return tuple(terminal[node_id] for node_id in sorted(terminal))

    def _require_authorities_unchanged(
        self, request: RunCaptureRequest, authorities: Mapping[str, Any]
    ) -> None:
        for payload_name, path in authorities["authority_paths"].items():
            if path.read_bytes() != authorities[payload_name]:
                raise CaptureExportError("capture authority changed during export")

    def _capture_branches(
        self,
        staging: Path,
        request: RunCaptureRequest,
        events: tuple[Any, ...],
        artifact_refs: dict[str, str],
        checksums: dict[str, str],
    ) -> tuple[list[str], dict[str, ArtifactCompleteness]]:
        branch_refs: list[str] = []
        completeness: dict[str, ArtifactCompleteness] = {}
        total_source_bytes = 0
        total_source_entries = 0
        for event in events:
            record = ExperimentRecord.from_dict(to_json_value(event.projection))
            key = f"branch:{record.node_id}:{record.execution_revision}"
            commit = self._resolve_pinned_commit(request.workspace_dir, event)
            if commit is None:
                if not record.had_error:
                    raise CaptureExportError(
                        "successful experiment branch is unavailable"
                    )
                completeness[key] = ArtifactCompleteness.UNAVAILABLE
                continue
            snapshot_root = (
                f"payload/branches/node-{record.node_id:08d}/"
                f"revision-{record.execution_revision:08d}"
            )
            source_files, excluded_files, source_bytes = self._export_source_tree(
                staging,
                request.workspace_dir,
                commit,
                snapshot_root,
                artifact_refs,
                checksums,
                total_source_entries,
                total_source_bytes,
            )
            total_source_entries += len(source_files) + len(excluded_files)
            total_source_bytes += source_bytes
            base_commit_shas = self._verify_recorded_base_commits(
                request.workspace_dir,
                commit,
                {
                    "implementation": event.artifact_refs.get(
                        "implementation_base", ""
                    ),
                    "diff": event.artifact_refs.get("diff_base", ""),
                    "feedback": event.artifact_refs.get("feedback_base", ""),
                },
                {
                    name: event.artifact_refs[f"{name}_base_commit"]
                    for name in ("implementation", "diff", "feedback")
                    if f"{name}_base_commit" in event.artifact_refs
                },
            )
            commit_objects, root_tree_sha = self._export_commit_objects(
                staging,
                request.workspace_dir,
                commit,
                base_commit_shas,
                snapshot_root,
                artifact_refs,
                checksums,
            )
            tree_descriptor = {
                item["source_path"]: (
                    item["sha256"],
                    item["mode"],
                    item["size"],
                )
                for item in source_files
            }
            tree_digest = (
                source_tree_digest(tree_descriptor)
                if tree_descriptor
                else tree_or_blob_digest(canonical_json_bytes(()))
            )
            snapshot = BranchSnapshot(
                schema=BRANCH_SNAPSHOT_SCHEMA,
                node_id=record.node_id,
                execution_revision=record.execution_revision,
                branch_name=record.branch_name,
                parent_branch_name=event.artifact_refs.get("parent_branch", ""),
                revision_ref=event.artifact_refs["candidate_ref"],
                commit_sha=commit,
                implementation_base_ref=event.artifact_refs.get(
                    "implementation_base", ""
                ),
                diff_base_ref=event.artifact_refs.get("diff_base", ""),
                feedback_base_ref=event.artifact_refs.get("feedback_base", ""),
                base_commit_shas=base_commit_shas,
                evaluated_commit_shas=tuple(
                    sorted(
                        {attempt.commit_sha for attempt in record.evaluation_attempts}
                    )
                ),
                root_tree_sha=root_tree_sha,
                commit_objects=tuple(commit_objects),
                source_tree_digest=tree_digest,
                source_files=tuple(source_files),
                excluded_files=tuple(excluded_files),
            )
            manifest_ref = f"{snapshot_root}/manifest.json"
            payload = snapshot.to_json_bytes()
            _write_bytes(staging / manifest_ref, payload)
            artifact_refs[key] = manifest_ref
            checksums[manifest_ref] = tree_or_blob_digest(payload)
            branch_refs.append(manifest_ref)
            completeness[key] = ArtifactCompleteness.PRESENT
        return branch_refs, completeness

    def _export_source_tree(
        self,
        staging: Path,
        workspace: Path,
        commit: str,
        snapshot_root: str,
        artifact_refs: dict[str, str],
        checksums: dict[str, str],
        prior_entries: int,
        prior_bytes: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        listing = self.git_command.run(
            workspace,
            ("ls-tree", "-r", "-l", "-z", commit),
            output_kind=CommandOutputKind.BINARY,
        )
        if listing.returncode != 0:
            raise CaptureExportError(listing.stderr.decode("utf-8", errors="strict"))
        entries = tuple(item for item in listing.stdout.split(b"\0") if item)
        if prior_entries + len(entries) > self.capture_settings.source_entry_limit:
            raise CaptureExportError("captured source entry limit exceeded")
        source_files: list[dict[str, Any]] = []
        excluded_files: list[dict[str, Any]] = []
        captured_bytes = 0
        for entry in entries:
            metadata, raw_path = entry.split(b"\t", 1)
            mode, object_type, object_sha, raw_size = metadata.decode("ascii").split()
            source_path = raw_path.decode("utf-8", errors="strict")
            normalized = PurePosixPath(source_path)
            if (
                normalized.is_absolute()
                or ".." in normalized.parts
                or normalized.as_posix() != source_path
            ):
                raise CaptureExportError("Git source tree contains an unsafe path")
            reason = self._source_exclusion_reason(
                source_path,
                mode,
                object_type,
                raw_size,
            )
            if reason is not None:
                excluded_files.append(
                    {
                        "git_object_sha": object_sha,
                        "mode": mode,
                        "object_type": object_type,
                        "path": source_path,
                        "reason": reason,
                        "size": None if raw_size == "-" else int(raw_size),
                    }
                )
                continue
            size = int(raw_size)
            if (
                prior_bytes + captured_bytes + size
                > self.capture_settings.bundle_asset_size_bytes
            ):
                raise CaptureExportError("captured source byte limit exceeded")
            blob = self.git_command.run(
                workspace,
                ("cat-file", "blob", object_sha),
                output_kind=CommandOutputKind.BINARY,
            )
            if blob.returncode != 0:
                raise CaptureExportError(blob.stderr.decode("utf-8", errors="strict"))
            if (
                len(blob.stdout) != size
                or git_object_sha("blob", blob.stdout) != object_sha
            ):
                raise CaptureExportError(
                    "Git blob content does not match its tree entry"
                )
            payload_ref = f"{snapshot_root}/files/{source_path}"
            _write_bytes(staging / payload_ref, blob.stdout)
            digest = tree_or_blob_digest(blob.stdout)
            artifact_refs[f"source:{len(artifact_refs):08d}"] = payload_ref
            checksums[payload_ref] = digest
            source_files.append(
                {
                    "git_blob_sha": object_sha,
                    "mode": mode,
                    "payload_ref": payload_ref,
                    "sha256": digest,
                    "size": size,
                    "source_path": source_path,
                }
            )
            captured_bytes += size
        return source_files, excluded_files, captured_bytes

    def _verify_recorded_base_commits(
        self,
        workspace: Path,
        candidate_commit: str,
        base_refs: Mapping[str, str],
        recorded_commits: Mapping[str, str],
    ) -> dict[str, str]:
        expected_names = {name for name, base_ref in base_refs.items() if base_ref}
        if set(recorded_commits) != expected_names:
            raise CaptureExportError("recorded base commit closure is incomplete")
        for name, commit in recorded_commits.items():
            if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
                raise CaptureExportError(f"recorded {name} base commit is invalid")
            exists = self.git_command.run(
                workspace,
                ("cat-file", "-e", f"{commit}^{{commit}}"),
                output_kind=CommandOutputKind.BINARY,
            )
            if exists.returncode != 0:
                raise CaptureExportError(f"recorded {name} base commit is missing")
            ancestry = self.git_command.run(
                workspace,
                (
                    "merge-base",
                    "--is-ancestor",
                    commit,
                    candidate_commit,
                ),
                output_kind=CommandOutputKind.BINARY,
            )
            if ancestry.returncode == 1:
                raise CaptureExportError(f"{name} base is not a candidate ancestor")
            if ancestry.returncode != 0:
                raise CaptureExportError(
                    ancestry.stderr.decode("utf-8", errors="strict")
                )
        return dict(recorded_commits)

    def _export_commit_objects(
        self,
        staging: Path,
        workspace: Path,
        candidate_commit: str,
        base_commit_shas: Mapping[str, str],
        snapshot_root: str,
        artifact_refs: dict[str, str],
        checksums: dict[str, str],
    ) -> tuple[list[dict[str, str]], str]:
        commit_shas = {candidate_commit, *base_commit_shas.values()}
        for base_commit in base_commit_shas.values():
            ancestry = self.git_command.run(
                workspace,
                (
                    "rev-list",
                    "--ancestry-path",
                    f"{base_commit}..{candidate_commit}",
                ),
                output_kind=CommandOutputKind.TEXT,
            )
            if ancestry.returncode != 0:
                raise CaptureExportError(
                    ancestry.stderr.decode("utf-8").strip()
                    or "could not export Git ancestry proof"
                )
            commit_shas.update(ancestry.output.splitlines())
        commit_objects: list[dict[str, str]] = []
        root_tree_sha = ""
        for commit_sha in sorted(commit_shas):
            if re.fullmatch(r"[0-9a-f]{40}", commit_sha) is None:
                raise CaptureExportError("Git ancestry returned an invalid commit")
            result = self.git_command.run(
                workspace,
                ("cat-file", "commit", commit_sha),
                output_kind=CommandOutputKind.BINARY,
            )
            if result.returncode != 0:
                raise CaptureExportError(result.stderr.decode("utf-8", errors="strict"))
            if git_object_sha("commit", result.stdout) != commit_sha:
                raise CaptureExportError("Git commit payload identity changed")
            parsed = parse_commit_object(result.stdout)
            if commit_sha == candidate_commit:
                root_tree_sha = parsed.tree_sha
            payload_ref = f"{snapshot_root}/commits/{commit_sha}.txt"
            _write_bytes(staging / payload_ref, result.stdout)
            artifact_refs[f"git_commit:{len(artifact_refs):08d}"] = payload_ref
            checksums[payload_ref] = tree_or_blob_digest(result.stdout)
            commit_objects.append(
                {"commit_sha": commit_sha, "payload_ref": payload_ref}
            )
        if not root_tree_sha:
            raise CaptureExportError("candidate Git commit evidence is missing")
        return commit_objects, root_tree_sha

    def _source_exclusion_reason(
        self,
        path: str,
        mode: str,
        object_type: str,
        raw_size: str,
    ) -> str | None:
        if object_type != "blob" or mode not in {"100644", "100755"}:
            return "non_regular_file"
        if raw_size == "-":
            return "unknown_size"
        size = int(raw_size)
        if size > self.sanitation_settings.max_file_bytes:
            return "file_too_large"
        if path_matches_denied_pattern(
            path, self.sanitation_settings.denied_path_patterns
        ):
            return "denied_path"
        source_name = PurePosixPath(path).name
        if (
            PurePosixPath(path).suffix.casefold()
            not in self.sanitation_settings.allowed_suffixes
            and source_name not in self.sanitation_settings.allowed_filenames
        ):
            return "artifact_class"
        return None

    def _resolve_pinned_commit(self, workspace: Path, event: Any) -> str | None:
        commit = event.artifact_refs.get("candidate_commit")
        pinned_ref = event.artifact_refs.get("candidate_ref")
        if commit is None:
            if pinned_ref is not None:
                raise CaptureExportError("revision pin has no candidate commit")
            return None
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise CaptureExportError("recorded candidate commit is invalid")
        expected_ref = (
            f"refs/kapso/execution-revisions/{event.run_id}/"
            f"node-{event.node_id}/revision-{event.execution_revision}"
        )
        if pinned_ref != expected_ref:
            raise CaptureExportError("execution revision immutable ref changed")
        require_git_ref_name(
            pinned_ref,
            "execution revision ref",
            qualified=True,
            error_type=CaptureExportError,
        )
        result = self.git_command.run(
            workspace,
            (
                "for-each-ref",
                "--format=%(objectname)",
                pinned_ref,
            ),
            output_kind=CommandOutputKind.TEXT,
        )
        if result.returncode != 0:
            raise CaptureExportError(
                result.stderr.decode("utf-8").strip()
                or "could not resolve execution revision ref"
            )
        resolved_commit = result.output.strip()
        if not resolved_commit:
            raise CaptureExportError("recorded execution revision ref is missing")
        if resolved_commit != commit:
            raise CaptureExportError("execution revision ref points to another commit")
        verify = self.git_command.run(
            workspace,
            ("cat-file", "-e", f"{commit}^{{commit}}"),
            output_kind=CommandOutputKind.BINARY,
        )
        if verify.returncode != 0:
            raise CaptureExportError("recorded execution revision commit is missing")
        return commit

    def _capture_run_logs(
        self,
        staging: Path,
        request: RunCaptureRequest,
        artifact_refs: dict[str, str],
        checksums: dict[str, str],
    ) -> list[str]:
        refs = []
        workspace = request.workspace_dir
        for position, relative in enumerate(request.run_log_paths):
            if path_matches_denied_pattern(
                relative, self.sanitation_settings.denied_path_patterns
            ):
                raise CaptureExportError("run log path is denied by sanitation policy")
            source_name = PurePosixPath(relative).name
            if (
                PurePosixPath(relative).suffix.casefold()
                not in self.sanitation_settings.allowed_suffixes
                and source_name not in self.sanitation_settings.allowed_filenames
            ):
                raise CaptureExportError("run log artifact class is not allowlisted")
            payload = read_restricted_regular_file(
                workspace,
                relative,
                CaptureExportError,
                require_restricted=False,
            )
            if len(payload) > self.sanitation_settings.max_file_bytes:
                raise CaptureExportError("run log exceeds sanitation file limit")
            destination = f"payload/run_logs/{position:08d}.txt"
            _write_bytes(staging / destination, payload)
            artifact_refs[f"run_log:{position:08d}"] = destination
            checksums[destination] = tree_or_blob_digest(payload)
            refs.append(destination)
        return refs

    def _load_current(
        self,
        run_root: Path,
        request: RunCaptureRequest,
    ) -> ExportedCapture | None:
        marker_path = run_root / CAPTURE_CURRENT_FILENAME
        if not marker_path.is_file():
            return None
        marker = parse_json_bytes(marker_path.read_bytes())
        if not isinstance(marker, dict) or set(marker) != {
            "capture_manifest_id",
            "generation",
            "path",
        }:
            raise CaptureExportError("capture current marker is invalid")
        if type(marker["generation"]) is not int or marker["generation"] < 0:
            raise CaptureExportError("capture current generation is invalid")
        expected_name = f"generation-{marker['generation']:020d}"
        if marker["path"] != expected_name:
            raise CaptureExportError("capture current path is invalid")
        capture_path = run_root / expected_name
        manifest = CaptureManifest.from_json_bytes(
            (capture_path / CAPTURE_MANIFEST_FILENAME).read_bytes()
        )
        descriptor = CaptureDescriptor.from_json_bytes(
            (capture_path / manifest.artifact_refs["capture_descriptor"]).read_bytes()
        )
        shared_fields = (
            "scope_contract_id",
            "scope_id",
            "run_id",
            "campaign_id",
            "capture_generation",
            "captured_at",
        )
        if any(
            getattr(manifest, field) != getattr(descriptor, field)
            for field in shared_fields
        ) or dict(manifest.artifact_refs) != dict(descriptor.artifact_refs):
            raise CaptureExportError(
                "capture current manifest and descriptor identities differ"
            )
        expected_identity = (
            request.scope_contract_id,
            request.scope_id,
            request.run_id,
            request.campaign_id,
            request.started_at,
            request.kapso_commit,
            request.launch_manifest_id,
            request.knowledge_snapshot_id,
            request.expert_base_release_id,
            request.task_context_binding,
            request.artifact_environment,
        )
        current_identity = (
            manifest.scope_contract_id,
            manifest.scope_id,
            manifest.run_id,
            manifest.campaign_id,
            descriptor.started_at,
            descriptor.kapso_commit,
            descriptor.launch_manifest_id,
            descriptor.knowledge_snapshot_id,
            descriptor.expert_base_release_id,
            descriptor.task_context_binding,
            descriptor.artifact_environment,
        )
        if current_identity != expected_identity:
            raise CaptureExportError("capture current belongs to another run identity")
        if (
            marker["capture_manifest_id"] != manifest.capture_manifest_id
            or marker["generation"] != manifest.capture_generation
        ):
            raise CaptureExportError("capture current marker conflicts with generation")
        if set(manifest.artifact_refs.values()) != set(manifest.checksums):
            raise CaptureExportError("current capture checksum closure is invalid")
        expected_files = {
            CAPTURE_MANIFEST_FILENAME,
            *manifest.artifact_refs.values(),
        }
        actual_files = set()
        for path in capture_path.rglob("*"):
            if path.is_symlink() or (not path.is_file() and not path.is_dir()):
                raise CaptureExportError("current capture contains an unsafe file")
            if path.is_file():
                actual_files.add(path.relative_to(capture_path).as_posix())
        if actual_files != expected_files:
            raise CaptureExportError("current capture file closure changed")
        for relative_path, expected_digest in manifest.checksums.items():
            payload_path = capture_path / relative_path
            if not payload_path.is_file() or payload_path.is_symlink():
                raise CaptureExportError("current capture artifact is missing")
            if tree_or_blob_digest(payload_path.read_bytes()) != expected_digest:
                raise CaptureExportError("current capture artifact digest changed")
        return ExportedCapture(capture_path, manifest, descriptor)

    @staticmethod
    def _remove_uncommitted_generations(
        run_root: Path, current: ExportedCapture | None
    ) -> None:
        current_generation = (
            current.manifest.capture_generation if current is not None else -1
        )
        for path in run_root.iterdir():
            if path.is_symlink():
                raise CaptureExportError("quarantine contains a symlink")
            if path.is_dir() and path.name.startswith(".capture."):
                shutil.rmtree(path)
                continue
            if not path.is_dir() or not path.name.startswith("generation-"):
                continue
            suffix = path.name.removeprefix("generation-")
            if not suffix.isdigit() or len(suffix) != 20:
                raise CaptureExportError("quarantine contains an invalid generation")
            if int(suffix) > current_generation:
                shutil.rmtree(path)

    @staticmethod
    def _same_frontier(
        current: ExportedCapture,
        proposed_manifest: CaptureManifest,
        proposed_descriptor: CaptureDescriptor,
    ) -> bool:
        current_checksums = dict(current.manifest.checksums)
        proposed_checksums = dict(proposed_manifest.checksums)
        current_checksums.pop(CAPTURE_DESCRIPTOR_REF)
        proposed_checksums.pop(CAPTURE_DESCRIPTOR_REF)
        current_descriptor = current.descriptor.to_dict()
        proposed_descriptor_values = proposed_descriptor.to_dict()
        for name in ("capture_generation", "captured_at"):
            current_descriptor.pop(name)
            proposed_descriptor_values.pop(name)
        return (
            current.manifest.scope_contract_id == proposed_manifest.scope_contract_id
            and current.manifest.scope_id == proposed_manifest.scope_id
            and current.manifest.run_id == proposed_manifest.run_id
            and current.manifest.campaign_id == proposed_manifest.campaign_id
            and current.manifest.checkpoint_frontier
            == proposed_manifest.checkpoint_frontier
            and current.manifest.capture_watermarks
            == proposed_manifest.capture_watermarks
            and current.manifest.configuration_fingerprint
            == proposed_manifest.configuration_fingerprint
            and current.manifest.artifact_refs == proposed_manifest.artifact_refs
            and current_checksums == proposed_checksums
            and current_descriptor == proposed_descriptor_values
        )
