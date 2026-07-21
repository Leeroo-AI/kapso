"""Immutable local content-addressed RunBundle handoff store."""

from __future__ import annotations

import fcntl
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    parse_json_bytes,
    require_content_id,
    tree_or_blob_digest,
)
from kapso.cross_run.capture.sanitation import (
    SANITATION_REPORT_REF,
    SanitizedCapture,
    SanitationReport,
    sanitation_policy_fingerprint,
)
from kapso.cross_run.capture.safety import (
    read_restricted_regular_file,
    remove_restricted_directory,
    restricted_directory_identity,
)
from kapso.cross_run.capture.validator import ValidatedCapture
from kapso.cross_run.contracts import RunBundle
from kapso.cross_run.settings import CaptureSettings, SanitationSettings

BUNDLE_MANIFEST_FILENAME = "manifest.json"
BUNDLE_REFS_FILENAME = "refs.json"
BUNDLE_CURRENT_FILENAME = "current.json"


class RunBundlePublicationError(ValueError):
    """A bundle conflicts with immutable local handoff state."""


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    os.fsync(descriptor)
    os.close(descriptor)


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o600)


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.chmod(0o600)
    os.replace(temporary, path)
    path.chmod(0o600)
    _fsync_directory(path.parent)


def _bundle_key(bundle_id: str) -> str:
    require_content_id(bundle_id, "bundle_id")
    return bundle_id.rsplit(":", 1)[-1]


def _run_key(run_id: str) -> str:
    return tree_or_blob_digest(run_id.encode("utf-8"))[7:]


def _reject_symlink_components(path: Path) -> None:
    for component in (path, *path.parents):
        if component.is_symlink():
            raise RunBundlePublicationError("bundle storage path contains a symlink")


@dataclass(frozen=True)
class StoredRunBundle:
    path: Path
    manifest: RunBundle
    object_refs: Mapping[str, str]

    def read_ref(self, relative_path: str) -> bytes:
        digest = self.object_refs.get(relative_path)
        if digest is None:
            raise RunBundlePublicationError(f"bundle has no ref: {relative_path}")
        object_path = self.path.parent.parent / "objects" / "sha256" / digest[7:]
        payload = read_restricted_regular_file(
            object_path.parent,
            object_path.name,
            RunBundlePublicationError,
        )
        if tree_or_blob_digest(payload) != digest:
            raise RunBundlePublicationError("bundle object digest changed")
        return payload


class RunBundlePublisher:
    """Publish one sanitized capture without GitHub calls or interpretation."""

    def __init__(
        self,
        store_root: str | Path,
        settings: CaptureSettings,
        sanitation_settings: SanitationSettings,
    ):
        self.settings = settings
        self.sanitation_settings = sanitation_settings
        self.root = Path(os.path.abspath(store_root))
        state_path = Path(settings.state_path)
        quarantine_path = Path(settings.quarantine_path)
        if state_path.is_absolute() or quarantine_path.is_absolute():
            raise RunBundlePublicationError(
                "capture storage paths must be workspace relative"
            )
        expected_store_tail = (*state_path.parts, "bundles")
        if self.root.parts[-len(expected_store_tail) :] != expected_store_tail:
            raise RunBundlePublicationError(
                "bundle store is outside the configured workspace state path"
            )
        self.workspace_root = Path(*self.root.parts[: -len(expected_store_tail)])
        self.state_root = self.workspace_root / state_path
        self.sanitized_root = self.state_root / "sanitized"
        self.quarantine_root = self.workspace_root / quarantine_path
        _reject_symlink_components(self.root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)

    def publish(
        self,
        capture: ValidatedCapture,
        sanitized: SanitizedCapture,
    ) -> StoredRunBundle:
        self._validate_publication_paths(capture, sanitized)
        self._validate_sanitized_capture(capture, sanitized)
        lock_path = self.root / ".publication.lock"
        _reject_symlink_components(lock_path)
        with lock_path.open("a+", encoding="utf-8") as lock_handle:
            lock_path.chmod(0o600)
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            current = self._load_current(capture)
            if current is not None:
                if (
                    current.manifest.capture_generation
                    > capture.manifest.capture_generation
                ):
                    raise RunBundlePublicationError("capture generation is stale")
                if (
                    current.manifest.capture_generation
                    == capture.manifest.capture_generation
                ):
                    if (
                        dict(current.manifest.checksums) != dict(sanitized.checksums)
                        or current.manifest.completion_state
                        is not capture.descriptor.completion_state
                    ):
                        raise RunBundlePublicationError(
                            "same capture generation produced different sanitized content"
                        )
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                    self._remove_sanitized_capture(sanitized)
                    self._prune_quarantine(capture)
                    return current
            descriptor = capture.descriptor
            manifest = RunBundle.mint(
                scope_contract_id=capture.manifest.scope_contract_id,
                scope_id=capture.manifest.scope_id,
                run_id=capture.manifest.run_id,
                campaign_id=capture.manifest.campaign_id,
                completion_state=descriptor.completion_state,
                capture_generation=capture.manifest.capture_generation,
                supersedes_bundle_id=(
                    current.manifest.bundle_id if current is not None else None
                ),
                checkpoint_frontier=capture.manifest.checkpoint_frontier,
                capture_watermarks=capture.manifest.capture_watermarks,
                configuration_fingerprint=capture.manifest.configuration_fingerprint,
                artifact_completeness=descriptor.artifact_completeness,
                started_at=descriptor.started_at,
                captured_at=descriptor.captured_at,
                kapso_commit=descriptor.kapso_commit,
                launch_manifest_id=descriptor.launch_manifest_id,
                knowledge_snapshot_id=descriptor.knowledge_snapshot_id,
                expert_base_release_id=descriptor.expert_base_release_id,
                task_context_binding=descriptor.task_context_binding,
                artifact_environment=descriptor.artifact_environment,
                checkpoint_ref=descriptor.artifact_refs["checkpoint"],
                execution_event_journal_ref=descriptor.artifact_refs[
                    "execution_event_journal"
                ],
                idea_archive_ref=descriptor.artifact_refs["idea_archive"],
                experiment_history_ref=descriptor.artifact_refs["experiment_history"],
                branch_snapshot_refs=descriptor.branch_snapshot_refs,
                run_log_refs=descriptor.run_log_refs,
                checksums=sanitized.checksums,
            )
            self._write_objects(sanitized)
            stored = self._commit_bundle(manifest, sanitized.checksums)
            marker = (
                canonical_json_bytes(
                    {
                        "bundle_id": manifest.bundle_id,
                        "capture_generation": manifest.capture_generation,
                    }
                )
                + b"\n"
            )
            runs_root = self.root / "runs"
            runs_root.mkdir(exist_ok=True, mode=0o700)
            runs_root.chmod(0o700)
            run_root = runs_root / _run_key(manifest.run_id)
            run_root.mkdir(exist_ok=True, mode=0o700)
            run_root.chmod(0o700)
            _write_atomic(
                run_root / BUNDLE_CURRENT_FILENAME,
                marker,
            )
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        self._remove_sanitized_capture(sanitized)
        self._prune_quarantine(capture)
        return stored

    def load(self, bundle_id: str) -> StoredRunBundle:
        path = self.root / "bundles" / _bundle_key(bundle_id)
        if not path.is_dir() or path.is_symlink():
            raise RunBundlePublicationError("bundle is absent from local store")
        manifest = RunBundle.from_json_bytes(
            read_restricted_regular_file(
                path,
                BUNDLE_MANIFEST_FILENAME,
                RunBundlePublicationError,
            )
        )
        refs = parse_json_bytes(
            read_restricted_regular_file(
                path,
                BUNDLE_REFS_FILENAME,
                RunBundlePublicationError,
            )
        )
        if manifest.bundle_id != bundle_id:
            raise RunBundlePublicationError("bundle directory identity changed")
        if not isinstance(refs, dict) or refs != dict(manifest.checksums):
            raise RunBundlePublicationError("bundle object refs are invalid")
        stored = StoredRunBundle(path=path, manifest=manifest, object_refs=refs)
        for relative_path in refs:
            stored.read_ref(relative_path)
        return stored

    def _load_current(self, capture: ValidatedCapture) -> StoredRunBundle | None:
        marker_path = (
            self.root
            / "runs"
            / _run_key(capture.manifest.run_id)
            / BUNDLE_CURRENT_FILENAME
        )
        if not marker_path.is_file():
            return None
        marker = parse_json_bytes(
            read_restricted_regular_file(
                marker_path.parent,
                marker_path.name,
                RunBundlePublicationError,
            )
        )
        if not isinstance(marker, dict) or set(marker) != {
            "bundle_id",
            "capture_generation",
        }:
            raise RunBundlePublicationError("bundle current marker is invalid")
        stored = self.load(marker["bundle_id"])
        if stored.manifest.capture_generation != marker["capture_generation"]:
            raise RunBundlePublicationError("bundle current marker generation changed")
        expected_identity = (
            capture.manifest.scope_contract_id,
            capture.manifest.scope_id,
            capture.manifest.run_id,
            capture.manifest.campaign_id,
            capture.descriptor.started_at,
            capture.descriptor.kapso_commit,
            capture.descriptor.launch_manifest_id,
            capture.descriptor.knowledge_snapshot_id,
            capture.descriptor.expert_base_release_id,
            capture.descriptor.task_context_binding,
            capture.descriptor.artifact_environment,
        )
        current_identity = (
            stored.manifest.scope_contract_id,
            stored.manifest.scope_id,
            stored.manifest.run_id,
            stored.manifest.campaign_id,
            stored.manifest.started_at,
            stored.manifest.kapso_commit,
            stored.manifest.launch_manifest_id,
            stored.manifest.knowledge_snapshot_id,
            stored.manifest.expert_base_release_id,
            stored.manifest.task_context_binding,
            stored.manifest.artifact_environment,
        )
        if current_identity != expected_identity:
            raise RunBundlePublicationError(
                "bundle current belongs to another run identity"
            )
        return stored

    def _write_objects(self, sanitized: SanitizedCapture) -> None:
        objects_root = self.root / "objects"
        objects_root.mkdir(exist_ok=True, mode=0o700)
        objects_root.chmod(0o700)
        object_root = objects_root / "sha256"
        object_root.mkdir(exist_ok=True, mode=0o700)
        object_root.chmod(0o700)
        for relative_path, digest in sorted(sanitized.checksums.items()):
            payload = read_restricted_regular_file(
                sanitized.path,
                relative_path,
                RunBundlePublicationError,
            )
            if tree_or_blob_digest(payload) != digest:
                raise RunBundlePublicationError("sanitized payload digest changed")
            object_path = object_root / digest[7:]
            if object_path.is_file():
                if (
                    read_restricted_regular_file(
                        object_root,
                        object_path.name,
                        RunBundlePublicationError,
                    )
                    != payload
                ):
                    raise RunBundlePublicationError(
                        "content-addressed object collision"
                    )
                continue
            _write_atomic(object_path, payload)

    def _validate_sanitized_capture(
        self,
        capture: ValidatedCapture,
        sanitized: SanitizedCapture,
    ) -> None:
        if sanitized.report.capture_manifest_id != capture.manifest.capture_manifest_id:
            raise RunBundlePublicationError(
                "sanitation report belongs to another capture"
            )
        if sanitized.report.status != "admitted":
            raise RunBundlePublicationError(
                "rejected sanitation result cannot be bundled"
            )
        if sanitized.report.policy_version != self.sanitation_settings.policy_version:
            raise RunBundlePublicationError("sanitation policy version changed")
        if sanitized.report.policy_fingerprint != sanitation_policy_fingerprint(
            self.sanitation_settings
        ):
            raise RunBundlePublicationError("sanitation policy settings changed")
        if dict(sanitized.artifact_refs) != dict(capture.manifest.artifact_refs):
            raise RunBundlePublicationError("sanitized artifact refs changed")
        required_paths = {
            *capture.manifest.artifact_refs.values(),
            SANITATION_REPORT_REF,
        }
        if set(sanitized.checksums) != required_paths:
            raise RunBundlePublicationError("sanitized checksum closure is not exact")
        admitted = dict(sanitized.checksums)
        admitted.pop(SANITATION_REPORT_REF)
        if dict(sanitized.report.admitted_refs) != admitted:
            raise RunBundlePublicationError(
                "sanitation report admission closure changed"
            )
        actual_files: set[str] = set()
        if not sanitized.path.is_dir() or sanitized.path.is_symlink():
            raise RunBundlePublicationError("sanitized root is not a real directory")
        for path in sanitized.path.rglob("*"):
            relative = path.relative_to(sanitized.path).as_posix()
            if path.is_symlink():
                raise RunBundlePublicationError("sanitized closure contains a symlink")
            metadata = path.stat()
            if metadata.st_mode & 0o077:
                raise RunBundlePublicationError(
                    "sanitized closure is not access restricted"
                )
            if path.is_file():
                if metadata.st_nlink != 1:
                    raise RunBundlePublicationError(
                        "sanitized closure contains a hard link"
                    )
                actual_files.add(relative)
            elif not path.is_dir():
                raise RunBundlePublicationError(
                    "sanitized closure contains an unsupported file"
                )
        if actual_files != required_paths:
            raise RunBundlePublicationError("sanitized file closure is not exact")
        for relative_path, digest in sanitized.checksums.items():
            payload = read_restricted_regular_file(
                sanitized.path,
                relative_path,
                RunBundlePublicationError,
            )
            if tree_or_blob_digest(payload) != digest:
                raise RunBundlePublicationError("sanitized payload digest changed")
        report_payload = read_restricted_regular_file(
            sanitized.path,
            SANITATION_REPORT_REF,
            RunBundlePublicationError,
        )
        if SanitationReport.from_json_bytes(report_payload) != sanitized.report:
            raise RunBundlePublicationError("sanitation report bytes changed")

    def _commit_bundle(
        self, manifest: RunBundle, refs: Mapping[str, str]
    ) -> StoredRunBundle:
        bundle_root = self.root / "bundles"
        bundle_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        bundle_root.chmod(0o700)
        destination = bundle_root / _bundle_key(manifest.bundle_id)
        if destination.is_dir():
            return self.load(manifest.bundle_id)
        staging = Path(tempfile.mkdtemp(prefix=".bundle.", dir=bundle_root))
        staging.chmod(0o700)
        _write_bytes(staging / BUNDLE_MANIFEST_FILENAME, manifest.to_json_bytes())
        _write_bytes(
            staging / BUNDLE_REFS_FILENAME,
            canonical_json_bytes(dict(refs)),
        )
        _fsync_directory(staging)
        os.replace(staging, destination)
        _fsync_directory(bundle_root)
        return self.load(manifest.bundle_id)

    def _validate_publication_paths(
        self,
        capture: ValidatedCapture,
        sanitized: SanitizedCapture,
    ) -> None:
        generation_name = f"generation-{capture.manifest.capture_generation:020d}"
        expected_capture_path = (
            self.quarantine_root
            / "runs"
            / _run_key(capture.manifest.run_id)
            / generation_name
        )
        capture_path = Path(os.path.abspath(capture.path))
        if capture_path != expected_capture_path:
            raise RunBundlePublicationError(
                "capture path is outside its configured run generation"
            )
        sanitized_path = Path(os.path.abspath(sanitized.path))
        if (
            sanitized_path.parent != self.sanitized_root
            or not sanitized_path.name.startswith(".sanitized.")
        ):
            raise RunBundlePublicationError(
                "sanitized path is outside the configured sanitation staging root"
            )
        _reject_symlink_components(capture_path)
        _reject_symlink_components(sanitized_path)
        if (
            restricted_directory_identity(
                self.sanitized_root,
                sanitized.path.name,
                RunBundlePublicationError,
            )
            != sanitized.directory_identity
        ):
            raise RunBundlePublicationError("sanitized staging identity changed")

    def _remove_sanitized_capture(self, sanitized: SanitizedCapture) -> None:
        remove_restricted_directory(
            self.sanitized_root,
            sanitized.path.name,
            sanitized.directory_identity,
            RunBundlePublicationError,
        )

    def _prune_quarantine(self, capture: ValidatedCapture) -> None:
        run_root = self.quarantine_root / "runs" / _run_key(capture.manifest.run_id)
        _reject_symlink_components(run_root)
        generations = sorted(
            (
                (
                    path,
                    restricted_directory_identity(
                        run_root,
                        path.name,
                        RunBundlePublicationError,
                    ),
                )
                for path in run_root.iterdir()
                if path.is_dir()
                and not path.is_symlink()
                and re.fullmatch(r"generation-[0-9]{20}", path.name) is not None
            ),
            key=lambda item: item[0].name,
        )
        removable = generations[: -self.settings.quarantine_retention_generations]
        for path, identity in removable:
            if (
                path.parent != run_root
                or re.fullmatch(r"generation-[0-9]{20}", path.name) is None
            ):
                raise RunBundlePublicationError("unsafe quarantine retention target")
            remove_restricted_directory(
                run_root,
                path.name,
                identity,
                RunBundlePublicationError,
            )
