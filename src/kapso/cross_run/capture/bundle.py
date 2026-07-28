"""Immutable local content-addressed RunBundle handoff store."""

from __future__ import annotations

import fcntl
import math
import os
import re
import stat
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Protocol

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    parse_json_bytes,
    require_content_id,
    tree_or_blob_digest,
)
from kapso.cross_run.capture.sanitation import (
    SANITATION_REPORT_REF,
    SanitizedCapture,
    sanitation_policy_fingerprint,
)
from kapso.cross_run.capture.safety import (
    read_restricted_regular_file,
    remove_restricted_directory,
    restricted_directory_identity,
)
from kapso.cross_run.capture.exporter import (
    CAPTURE_CURRENT_FILENAME,
    CAPTURE_EXPORT_LOCK_FILENAME,
    CAPTURE_MANIFEST_FILENAME,
)
from kapso.cross_run.capture.validator import ValidatedCapture
from kapso.cross_run.contracts import (
    CaptureManifest,
    ExpertSourceReplayContextMaterializationReceipt,
    ExpertSourceReplayStartingArtifact,
    RunBundle,
    TaskContextBinding,
)
from kapso.cross_run.expert.replay_context import (
    VerifiedSourceReplayContext,
    VerifiedSourceReplayStartingArtifact,
)
from kapso.cross_run.expert.task_evaluation_materialization import (
    TaskEvaluationMaterializationLimits,
)
from kapso.cross_run.launch.resolver import VerifiedLaunchStartingArtifact
from kapso.cross_run.record_contracts import SanitationReport
from kapso.cross_run.settings import (
    CaptureSettings,
    ExpertValidationSettings,
    SanitationSettings,
)

BUNDLE_MANIFEST_FILENAME = "manifest.json"
BUNDLE_CURRENT_FILENAME = "current.json"
_STORE_DIRECTORY_MODE = 0o700
_IMMUTABLE_DIRECTORY_MODE = 0o500
_IMMUTABLE_FILE_MODE = 0o400
_MUTABLE_CONTROL_MODE = 0o600


class RunBundlePublicationError(ValueError):
    """A bundle conflicts with immutable local handoff state."""


@dataclass(frozen=True)
class _StoreIdentity:
    root: tuple[int, int]
    objects: tuple[int, int]
    object_payloads: tuple[int, int]
    bundles: tuple[int, int]
    starting_artifacts: tuple[int, int]
    runs: tuple[int, int]


def _descriptor_identity(descriptor: int) -> tuple[int, int]:
    metadata = os.fstat(descriptor)
    return metadata.st_dev, metadata.st_ino


def _require_descriptor_identity(
    descriptor: int,
    expected: tuple[int, int],
) -> None:
    if _descriptor_identity(descriptor) != expected:
        raise RunBundlePublicationError("bundle store directory identity changed")


def _require_child_directory_binding(
    parent_descriptor: int,
    name: str,
    child_descriptor: int,
) -> None:
    current = os.stat(
        name,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    if not stat.S_ISDIR(current.st_mode) or (
        current.st_dev,
        current.st_ino,
    ) != _descriptor_identity(child_descriptor):
        raise RunBundlePublicationError("bundle run directory binding changed")


def _bundle_key(bundle_id: str) -> str:
    require_content_id(bundle_id, "bundle_id")
    return bundle_id.rsplit(":", 1)[-1]


def _run_key(run_id: str) -> str:
    return tree_or_blob_digest(run_id.encode("utf-8"))[7:]


def _current_marker_payload(manifest: RunBundle) -> bytes:
    return (
        canonical_json_bytes(
            {
                "bundle_id": manifest.bundle_id,
                "capture_generation": manifest.capture_generation,
            }
        )
        + b"\n"
    )


def _parse_current_marker(marker_payload: bytes) -> Mapping[str, object]:
    marker = parse_json_bytes(marker_payload)
    if (
        not isinstance(marker, dict)
        or set(marker) != {"bundle_id", "capture_generation"}
        or not isinstance(marker["bundle_id"], str)
        or type(marker["capture_generation"]) is not int
        or marker["capture_generation"] < 0
        or marker_payload != canonical_json_bytes(marker) + b"\n"
    ):
        raise RunBundlePublicationError("bundle current marker is invalid")
    require_content_id(marker["bundle_id"], "bundle current ID")
    return marker


def _reject_symlink_components(path: Path) -> None:
    for component in (path, *path.parents):
        if component.is_symlink():
            raise RunBundlePublicationError("bundle storage path contains a symlink")


def _directory_names(
    descriptor: int,
    maximum_entries: int,
    label: str,
) -> tuple[str, ...]:
    names: list[str] = []
    with os.scandir(descriptor) as entries:
        for entry in entries:
            names.append(entry.name)
            if len(names) > maximum_entries:
                raise RunBundlePublicationError(
                    f"{label} exceeds configured entry limit"
                )
    return tuple(names)


def _open_absolute_directory(
    path: Path,
    descriptors: ExitStack,
    *,
    required_mode: int | None,
) -> int:
    absolute = Path(os.path.abspath(path))
    descriptor = os.open(
        absolute.anchor,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    descriptors.callback(os.close, descriptor)
    for name in absolute.parts[1:]:
        child = os.open(
            name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=descriptor,
        )
        descriptors.callback(os.close, child)
        descriptor = child
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode) or (
        required_mode is not None and stat.S_IMODE(metadata.st_mode) != required_mode
    ):
        raise RunBundlePublicationError("bundle store directory mode is invalid")
    return descriptor


def _open_child_directory(
    parent_descriptor: int,
    name: str,
    descriptors: ExitStack,
    *,
    mode: int,
    create: bool,
) -> int:
    exists = os.access(
        name,
        os.F_OK,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    if not exists:
        if not create:
            raise RunBundlePublicationError("bundle store layout is incomplete")
        os.mkdir(name, mode=mode, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=parent_descriptor,
    )
    descriptors.callback(os.close, descriptor)
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != mode:
        raise RunBundlePublicationError("bundle store directory mode is invalid")
    return descriptor


def _read_regular_file_at(
    parent_descriptor: int,
    name: str,
    *,
    maximum_bytes: int,
    required_mode: int,
) -> bytes:
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=parent_descriptor,
    )
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != required_mode
    ):
        os.close(descriptor)
        raise RunBundlePublicationError("bundle store file identity is invalid")
    if metadata.st_size > maximum_bytes:
        os.close(descriptor)
        raise RunBundlePublicationError("artifact exceeds configured size limit")
    with os.fdopen(descriptor, "rb") as handle:
        payload = handle.read(maximum_bytes + 1)
    if len(payload) > maximum_bytes:
        raise RunBundlePublicationError("artifact exceeds configured size limit")
    return payload


def _write_new_file_at(
    parent_descriptor: int,
    name: str,
    payload: bytes,
    *,
    mode: int,
) -> None:
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        mode,
        dir_fd=parent_descriptor,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fchmod(handle.fileno(), mode)
        os.fsync(handle.fileno())


def _write_atomic_file_at(
    parent_descriptor: int,
    name: str,
    payload: bytes,
    *,
    mode: int,
    maximum_staging_bytes: int,
) -> None:
    temporary_name = f".{name}.tmp"
    _discard_staged_regular_file_at(
        parent_descriptor,
        temporary_name,
        required_mode=mode,
        maximum_bytes=maximum_staging_bytes,
    )
    _write_new_file_at(
        parent_descriptor,
        temporary_name,
        payload,
        mode=mode,
    )
    os.replace(
        temporary_name,
        name,
        src_dir_fd=parent_descriptor,
        dst_dir_fd=parent_descriptor,
    )
    os.fsync(parent_descriptor)


def _discard_staged_regular_file_at(
    parent_descriptor: int,
    name: str,
    *,
    required_mode: int,
    maximum_bytes: int,
) -> None:
    if not os.access(
        name,
        os.F_OK,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    ):
        return
    _read_regular_file_at(
        parent_descriptor,
        name,
        maximum_bytes=maximum_bytes,
        required_mode=required_mode,
    )
    os.unlink(name, dir_fd=parent_descriptor)
    os.fsync(parent_descriptor)


@dataclass(frozen=True)
class StoredRunBundle:
    manifest: RunBundle
    artifacts: Mapping[str, bytes]

    def __post_init__(self) -> None:
        if set(self.artifacts) != set(self.manifest.checksums):
            raise RunBundlePublicationError("bundle artifact byte closure is not exact")
        if any(
            not isinstance(payload, bytes)
            or tree_or_blob_digest(payload) != self.manifest.checksums[relative_path]
            for relative_path, payload in self.artifacts.items()
        ):
            raise RunBundlePublicationError("bundle artifact digest changed")
        object.__setattr__(
            self,
            "artifacts",
            MappingProxyType(dict(self.artifacts)),
        )

    def read_ref(self, relative_path: str) -> bytes:
        payload = self.artifacts.get(relative_path)
        if payload is None:
            raise RunBundlePublicationError(f"bundle has no ref: {relative_path}")
        return payload


class RunBundleReader(Protocol):
    """Read one already snapshotted immutable bundle byte closure."""

    manifest: RunBundle

    def read_ref(self, relative_path: str) -> bytes: ...


class RunBundleStore:
    """Resolve exact bundle IDs without following mutable run pointers."""

    @classmethod
    def initialize(
        cls,
        state_root: str | Path,
        settings: CaptureSettings,
        sanitation_settings: SanitationSettings,
    ) -> RunBundleStore:
        """Create or reopen the single additive replay-evidence CAS."""

        root = Path(os.path.abspath(state_root))
        state_path = Path(settings.state_path)
        if state_path.is_absolute() or root.parts[-len(state_path.parts) :] != (
            state_path.parts
        ):
            raise RunBundlePublicationError(
                "bundle store is outside the configured workspace state path"
            )
        workspace_root = Path(*root.parts[: -len(state_path.parts)])
        if workspace_root in {Path("/"), Path.home()}:
            raise RunBundlePublicationError("bundle workspace root is unsafe")
        with ExitStack() as descriptors:
            state_descriptor = _open_absolute_directory(
                workspace_root,
                descriptors,
                required_mode=None,
            )
            bootstrap_descriptor = state_descriptor
            fcntl.flock(bootstrap_descriptor, fcntl.LOCK_EX)
            for name in state_path.parts:
                state_descriptor = _open_child_directory(
                    state_descriptor,
                    name,
                    descriptors,
                    mode=_STORE_DIRECTORY_MODE,
                    create=True,
                )
            objects_descriptor = _open_child_directory(
                state_descriptor,
                "objects",
                descriptors,
                mode=_STORE_DIRECTORY_MODE,
                create=True,
            )
            _open_child_directory(
                objects_descriptor,
                "sha256",
                descriptors,
                mode=_STORE_DIRECTORY_MODE,
                create=True,
            )
            for name in ("bundles", "starting_artifacts", "runs"):
                _open_child_directory(
                    state_descriptor,
                    name,
                    descriptors,
                    mode=_STORE_DIRECTORY_MODE,
                    create=True,
                )
            fcntl.flock(bootstrap_descriptor, fcntl.LOCK_UN)
        return cls(root, settings, sanitation_settings)

    def __init__(
        self,
        state_root: str | Path,
        settings: CaptureSettings,
        sanitation_settings: SanitationSettings,
    ) -> None:
        self.settings = settings
        self.sanitation_settings = sanitation_settings
        self.root = Path(os.path.abspath(state_root))
        state_path = Path(settings.state_path)
        if state_path.is_absolute() or self.root.parts[-len(state_path.parts) :] != (
            state_path.parts
        ):
            raise RunBundlePublicationError(
                "bundle store is outside the configured workspace state path"
            )
        self.workspace_root = Path(*self.root.parts[: -len(state_path.parts)])
        if self.workspace_root in {Path("/"), Path.home()}:
            raise RunBundlePublicationError("bundle workspace root is unsafe")
        with ExitStack() as descriptors:
            root_descriptor = _open_absolute_directory(
                self.root,
                descriptors,
                required_mode=_STORE_DIRECTORY_MODE,
            )
            objects_descriptor = _open_child_directory(
                root_descriptor,
                "objects",
                descriptors,
                mode=_STORE_DIRECTORY_MODE,
                create=False,
            )
            object_payload_descriptor = _open_child_directory(
                objects_descriptor,
                "sha256",
                descriptors,
                mode=_STORE_DIRECTORY_MODE,
                create=False,
            )
            bundles_descriptor = _open_child_directory(
                root_descriptor,
                "bundles",
                descriptors,
                mode=_STORE_DIRECTORY_MODE,
                create=False,
            )
            starting_artifacts_descriptor = _open_child_directory(
                root_descriptor,
                "starting_artifacts",
                descriptors,
                mode=_STORE_DIRECTORY_MODE,
                create=False,
            )
            runs_descriptor = _open_child_directory(
                root_descriptor,
                "runs",
                descriptors,
                mode=_STORE_DIRECTORY_MODE,
                create=False,
            )
            self.identity = _StoreIdentity(
                root=_descriptor_identity(root_descriptor),
                objects=_descriptor_identity(objects_descriptor),
                object_payloads=_descriptor_identity(object_payload_descriptor),
                bundles=_descriptor_identity(bundles_descriptor),
                starting_artifacts=_descriptor_identity(starting_artifacts_descriptor),
                runs=_descriptor_identity(runs_descriptor),
            )

    def read_manifest_exact(
        self,
        bundle_id: str,
        *,
        deadline: float | None = None,
    ) -> RunBundle | None:
        self._validate_deadline(deadline)
        with ExitStack() as descriptors:
            root_descriptor = _open_absolute_directory(
                self.root,
                descriptors,
                required_mode=_STORE_DIRECTORY_MODE,
            )
            _require_descriptor_identity(root_descriptor, self.identity.root)
            manifest = self._read_manifest_at(
                root_descriptor,
                descriptors,
                bundle_id,
                deadline,
            )
        self._require_deadline(deadline)
        return manifest

    def read_exact(self, bundle_id: str) -> StoredRunBundle | None:
        return self._read_exact(
            bundle_id,
            maximum_entries=self.settings.bundle_entry_limit,
            maximum_bytes=self.settings.bundle_asset_size_bytes,
            deadline=None,
            replay_bounded=False,
        )

    def read_exact_bounded(
        self,
        bundle_id: str,
        *,
        maximum_entries: int,
        maximum_bytes: int,
        deadline: float,
    ) -> StoredRunBundle | None:
        if (
            type(maximum_entries) is not int
            or maximum_entries <= 0
            or type(maximum_bytes) is not int
            or maximum_bytes <= 0
        ):
            raise RunBundlePublicationError(
                "bounded bundle limits must be positive integers"
            )
        self._validate_deadline(deadline)
        return self._read_exact(
            bundle_id,
            maximum_entries=min(
                maximum_entries,
                self.settings.bundle_entry_limit,
            ),
            maximum_bytes=min(
                maximum_bytes,
                self.settings.bundle_asset_size_bytes,
            ),
            deadline=deadline,
            replay_bounded=True,
        )

    def _read_exact(
        self,
        bundle_id: str,
        *,
        maximum_entries: int,
        maximum_bytes: int,
        deadline: float | None,
        replay_bounded: bool,
    ) -> StoredRunBundle | None:
        self._require_deadline(deadline)
        with ExitStack() as descriptors:
            root_descriptor = _open_absolute_directory(
                self.root,
                descriptors,
                required_mode=_STORE_DIRECTORY_MODE,
            )
            _require_descriptor_identity(root_descriptor, self.identity.root)
            manifest = self._read_manifest_at(
                root_descriptor,
                descriptors,
                bundle_id,
                deadline,
            )
            if manifest is None:
                return None
            if len(manifest.checksums) > maximum_entries:
                message = (
                    "bundle exceeds remaining replay materialization budget"
                    if replay_bounded
                    else "bundle artifact entry limit exceeded"
                )
                raise RunBundlePublicationError(message)
            objects_descriptor = _open_child_directory(
                root_descriptor,
                "objects",
                descriptors,
                mode=_STORE_DIRECTORY_MODE,
                create=False,
            )
            _require_descriptor_identity(
                objects_descriptor,
                self.identity.objects,
            )
            object_payload_descriptor = _open_child_directory(
                objects_descriptor,
                "sha256",
                descriptors,
                mode=_STORE_DIRECTORY_MODE,
                create=False,
            )
            _require_descriptor_identity(
                object_payload_descriptor,
                self.identity.object_payloads,
            )
            expected_bytes = self._preflight_object_closure(
                object_payload_descriptor,
                manifest,
                maximum_bytes,
                deadline,
                replay_bounded,
            )
            artifacts: dict[str, bytes] = {}
            total_bytes = 0
            for relative_path, digest in sorted(manifest.checksums.items()):
                self._require_deadline(deadline)
                payload = _read_regular_file_at(
                    object_payload_descriptor,
                    digest[7:],
                    maximum_bytes=self.sanitation_settings.max_file_bytes,
                    required_mode=_IMMUTABLE_FILE_MODE,
                )
                total_bytes += len(payload)
                if total_bytes > maximum_bytes:
                    message = (
                        "bundle exceeds remaining replay materialization budget"
                        if replay_bounded
                        else "bundle artifact byte limit exceeded"
                    )
                    raise RunBundlePublicationError(message)
                if tree_or_blob_digest(payload) != digest:
                    raise RunBundlePublicationError("bundle object digest changed")
                artifacts[relative_path] = payload
            if total_bytes != expected_bytes:
                raise RunBundlePublicationError(
                    "bundle object sizes changed during bounded acquisition"
                )
        self._require_deadline(deadline)
        report = SanitationReport.from_json_bytes(
            artifacts[manifest.sanitation_report_ref]
        )
        admitted = dict(manifest.checksums)
        admitted.pop(manifest.sanitation_report_ref)
        if (
            report.status != "admitted"
            or report.scope_id != manifest.scope_id
            or report.task_family_id != manifest.task_context_binding.task_family_id
            or dict(report.admitted_refs) != admitted
        ):
            raise RunBundlePublicationError(
                "sanitation report does not bind the bundle"
            )
        self._require_deadline(deadline)
        return StoredRunBundle(manifest=manifest, artifacts=artifacts)

    def _read_manifest_at(
        self,
        root_descriptor: int,
        descriptors: ExitStack,
        bundle_id: str,
        deadline: float | None,
    ) -> RunBundle | None:
        self._require_deadline(deadline)
        bundles_descriptor = _open_child_directory(
            root_descriptor,
            "bundles",
            descriptors,
            mode=_STORE_DIRECTORY_MODE,
            create=False,
        )
        _require_descriptor_identity(
            bundles_descriptor,
            self.identity.bundles,
        )
        bundle_key = _bundle_key(bundle_id)
        if not os.access(
            bundle_key,
            os.F_OK,
            dir_fd=bundles_descriptor,
            follow_symlinks=False,
        ):
            return None
        bundle_descriptor = _open_child_directory(
            bundles_descriptor,
            bundle_key,
            descriptors,
            mode=_IMMUTABLE_DIRECTORY_MODE,
            create=False,
        )
        if _directory_names(bundle_descriptor, 2, "bundle control closure") != (
            BUNDLE_MANIFEST_FILENAME,
        ):
            raise RunBundlePublicationError("bundle control closure is not exact")
        self._require_deadline(deadline)
        manifest_payload = _read_regular_file_at(
            bundle_descriptor,
            BUNDLE_MANIFEST_FILENAME,
            maximum_bytes=self.sanitation_settings.max_file_bytes,
            required_mode=_IMMUTABLE_FILE_MODE,
        )
        self._require_deadline(deadline)
        manifest = RunBundle.from_json_bytes(manifest_payload)
        if manifest.to_json_bytes() != manifest_payload:
            raise RunBundlePublicationError("bundle manifest is not canonical")
        if manifest.bundle_id != bundle_id:
            raise RunBundlePublicationError("bundle directory identity changed")
        if len(manifest.checksums) > self.settings.bundle_entry_limit:
            raise RunBundlePublicationError("bundle artifact entry limit exceeded")
        return manifest

    def _preflight_object_closure(
        self,
        object_payload_descriptor: int,
        manifest: RunBundle,
        maximum_bytes: int,
        deadline: float | None,
        replay_bounded: bool,
    ) -> int:
        total_bytes = 0
        for digest in manifest.checksums.values():
            self._require_deadline(deadline)
            metadata = os.stat(
                digest[7:],
                dir_fd=object_payload_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != _IMMUTABLE_FILE_MODE
            ):
                raise RunBundlePublicationError("bundle store file identity is invalid")
            total_bytes += metadata.st_size
            if total_bytes > maximum_bytes:
                message = (
                    "bundle exceeds remaining replay materialization budget"
                    if replay_bounded
                    else "bundle artifact byte limit exceeded"
                )
                raise RunBundlePublicationError(message)
        return total_bytes

    @staticmethod
    def _validate_deadline(deadline: float | None) -> None:
        if deadline is not None and (
            isinstance(deadline, bool)
            or not isinstance(deadline, (int, float))
            or not math.isfinite(float(deadline))
        ):
            raise RunBundlePublicationError(
                "bundle acquisition deadline must be finite"
            )

    @staticmethod
    def _require_deadline(deadline: float | None) -> None:
        if deadline is not None and time.monotonic() >= deadline:
            raise RunBundlePublicationError(
                "bundle replay materialization deadline expired"
            )

    def require_exact(self, bundle_id: str) -> StoredRunBundle:
        stored = self.read_exact(bundle_id)
        if stored is None:
            raise RunBundlePublicationError("bundle is absent from local store")
        return stored

    def import_exact(self, bundle: StoredRunBundle) -> StoredRunBundle:
        """Idempotently add one already verified sanitized bundle closure."""

        if type(bundle) is not StoredRunBundle:
            raise RunBundlePublicationError(
                "bundle import requires an exact stored bundle"
            )
        manifest = bundle.manifest
        report = SanitationReport.from_json_bytes(
            bundle.read_ref(manifest.sanitation_report_ref)
        )
        admitted = dict(manifest.checksums)
        admitted.pop(manifest.sanitation_report_ref)
        if (
            len(manifest.checksums) > self.settings.bundle_entry_limit
            or sum(len(payload) for payload in bundle.artifacts.values())
            > self.settings.bundle_asset_size_bytes
            or report.status != "admitted"
            or report.scope_id != manifest.scope_id
            or report.task_family_id != manifest.task_context_binding.task_family_id
            or dict(report.admitted_refs) != admitted
        ):
            raise RunBundlePublicationError(
                "imported bundle differs from its sanitation authority"
            )
        with ExitStack() as descriptors:
            root_descriptor = _open_absolute_directory(
                self.root,
                descriptors,
                required_mode=_STORE_DIRECTORY_MODE,
            )
            _require_descriptor_identity(root_descriptor, self.identity.root)
            fcntl.flock(root_descriptor, fcntl.LOCK_EX)
            objects_descriptor = _open_child_directory(
                root_descriptor,
                "objects",
                descriptors,
                mode=_STORE_DIRECTORY_MODE,
                create=False,
            )
            _require_descriptor_identity(objects_descriptor, self.identity.objects)
            object_payload_descriptor = _open_child_directory(
                objects_descriptor,
                "sha256",
                descriptors,
                mode=_STORE_DIRECTORY_MODE,
                create=False,
            )
            _require_descriptor_identity(
                object_payload_descriptor,
                self.identity.object_payloads,
            )
            bundles_descriptor = _open_child_directory(
                root_descriptor,
                "bundles",
                descriptors,
                mode=_STORE_DIRECTORY_MODE,
                create=False,
            )
            _require_descriptor_identity(bundles_descriptor, self.identity.bundles)
            self._write_object_payloads(bundle.artifacts, object_payload_descriptor)
            self._commit_manifest_directory(
                parent_descriptor=bundles_descriptor,
                content_key=_bundle_key(manifest.bundle_id),
                staging_prefix="bundle",
                payload=manifest.to_json_bytes(),
            )
            fcntl.flock(root_descriptor, fcntl.LOCK_UN)
        imported = self.require_exact(manifest.bundle_id)
        if imported != bundle:
            raise RunBundlePublicationError(
                "imported bundle differs from its exact source"
            )
        return imported

    def publish_starting_artifacts(
        self,
        *,
        task_context_binding: TaskContextBinding,
        launch_artifacts: tuple[VerifiedLaunchStartingArtifact, ...],
        validation_settings: ExpertValidationSettings,
    ) -> VerifiedSourceReplayContext:
        """Persist launch-verified task inputs under replay content identities."""

        if (
            type(task_context_binding) is not TaskContextBinding
            or type(launch_artifacts) is not tuple
            or any(
                type(item) is not VerifiedLaunchStartingArtifact
                for item in launch_artifacts
            )
            or type(validation_settings) is not ExpertValidationSettings
        ):
            raise RunBundlePublicationError(
                "starting-artifact publication requires exact launch authorities"
            )
        verified = tuple(
            VerifiedSourceReplayStartingArtifact(
                artifact=ExpertSourceReplayStartingArtifact.mint(
                    starting_artifact_ref=item.artifact.starting_artifact_ref,
                    mount_path=item.artifact.mount_path,
                    materialized_tree_hash=item.artifact.materialized_tree_hash,
                    source_files=item.artifact.source_files,
                ),
                source_contents=item.source_contents,
            )
            for item in launch_artifacts
        )
        verified = tuple(
            sorted(
                verified,
                key=lambda item: item.artifact.starting_artifact_content_id,
            )
        )
        refs_to_ids = {
            item.artifact.starting_artifact_ref: (
                item.artifact.starting_artifact_content_id
            )
            for item in verified
        }
        if set(refs_to_ids) != set(task_context_binding.starting_artifact_refs):
            raise RunBundlePublicationError(
                "starting artifacts differ from their task-context binding"
            )
        with ExitStack() as descriptors:
            root_descriptor = _open_absolute_directory(
                self.root,
                descriptors,
                required_mode=_STORE_DIRECTORY_MODE,
            )
            _require_descriptor_identity(root_descriptor, self.identity.root)
            fcntl.flock(root_descriptor, fcntl.LOCK_EX)
            objects_descriptor = _open_child_directory(
                root_descriptor,
                "objects",
                descriptors,
                mode=_STORE_DIRECTORY_MODE,
                create=False,
            )
            object_payload_descriptor = _open_child_directory(
                objects_descriptor,
                "sha256",
                descriptors,
                mode=_STORE_DIRECTORY_MODE,
                create=False,
            )
            starting_artifacts_descriptor = _open_child_directory(
                root_descriptor,
                "starting_artifacts",
                descriptors,
                mode=_STORE_DIRECTORY_MODE,
                create=False,
            )
            _require_descriptor_identity(
                object_payload_descriptor,
                self.identity.object_payloads,
            )
            _require_descriptor_identity(
                starting_artifacts_descriptor,
                self.identity.starting_artifacts,
            )
            for item in verified:
                payloads = {
                    descriptor.relative_path: item.source_contents[
                        descriptor.relative_path
                    ]
                    for descriptor in item.artifact.source_files
                }
                self._write_object_payloads(payloads, object_payload_descriptor)
                self._commit_manifest_directory(
                    parent_descriptor=starting_artifacts_descriptor,
                    content_key=_bundle_key(item.artifact.starting_artifact_content_id),
                    staging_prefix="starting-artifact",
                    payload=item.artifact.to_json_bytes(),
                )
            fcntl.flock(root_descriptor, fcntl.LOCK_UN)
        return self.materialize_exact(
            task_context_binding,
            refs_to_ids,
            TaskEvaluationMaterializationLimits(
                maximum_entries=(
                    validation_settings.policy.task_evaluation_materialization_entry_limit
                ),
                maximum_bytes=(
                    validation_settings.policy.task_evaluation_materialization_byte_limit
                ),
                timeout_seconds=(
                    validation_settings.policy.task_evaluation_materialization_timeout_seconds
                ),
            ),
            validation_settings=validation_settings,
        )

    def materialize_exact(
        self,
        task_context_binding: TaskContextBinding,
        expected_artifact_content_ids: Mapping[str, str],
        limits: TaskEvaluationMaterializationLimits,
        *,
        validation_settings: ExpertValidationSettings,
    ) -> VerifiedSourceReplayContext:
        """Resolve one exact captured task-input closure under remaining limits."""

        if (
            type(task_context_binding) is not TaskContextBinding
            or not isinstance(expected_artifact_content_ids, Mapping)
            or type(limits) is not TaskEvaluationMaterializationLimits
            or type(validation_settings) is not ExpertValidationSettings
        ):
            raise RunBundlePublicationError(
                "starting-artifact materialization requires exact authorities"
            )
        expected = dict(expected_artifact_content_ids)
        if set(expected) != set(task_context_binding.starting_artifact_refs):
            raise RunBundlePublicationError(
                "starting-artifact request differs from its task context"
            )
        deadline = time.monotonic() + limits.timeout_seconds
        verified = []
        entry_count = 0
        byte_count = 0
        with ExitStack() as descriptors:
            root_descriptor = _open_absolute_directory(
                self.root,
                descriptors,
                required_mode=_STORE_DIRECTORY_MODE,
            )
            _require_descriptor_identity(root_descriptor, self.identity.root)
            fcntl.flock(root_descriptor, fcntl.LOCK_SH)
            objects_descriptor = _open_child_directory(
                root_descriptor,
                "objects",
                descriptors,
                mode=_STORE_DIRECTORY_MODE,
                create=False,
            )
            object_payload_descriptor = _open_child_directory(
                objects_descriptor,
                "sha256",
                descriptors,
                mode=_STORE_DIRECTORY_MODE,
                create=False,
            )
            starting_artifacts_descriptor = _open_child_directory(
                root_descriptor,
                "starting_artifacts",
                descriptors,
                mode=_STORE_DIRECTORY_MODE,
                create=False,
            )
            _require_descriptor_identity(
                object_payload_descriptor,
                self.identity.object_payloads,
            )
            _require_descriptor_identity(
                starting_artifacts_descriptor,
                self.identity.starting_artifacts,
            )
            for artifact_ref, artifact_id in sorted(expected.items()):
                self._require_materialization_deadline(deadline)
                artifact = self._read_starting_artifact(
                    starting_artifacts_descriptor,
                    artifact_id,
                )
                if artifact.starting_artifact_ref != artifact_ref:
                    raise RunBundlePublicationError(
                        "starting-artifact reference differs from its identity"
                    )
                entry_count += len(artifact.source_files)
                if entry_count > limits.maximum_entries:
                    raise RunBundlePublicationError(
                        "starting artifacts exceed remaining entry budget"
                    )
                contents = {}
                for descriptor in artifact.source_files:
                    self._require_materialization_deadline(deadline)
                    payload = _read_regular_file_at(
                        object_payload_descriptor,
                        descriptor.digest[7:],
                        maximum_bytes=self.sanitation_settings.max_file_bytes,
                        required_mode=_IMMUTABLE_FILE_MODE,
                    )
                    byte_count += len(payload)
                    if byte_count > limits.maximum_bytes:
                        raise RunBundlePublicationError(
                            "starting artifacts exceed remaining byte budget"
                        )
                    contents[descriptor.relative_path] = payload
                verified.append(
                    VerifiedSourceReplayStartingArtifact(
                        artifact=artifact,
                        source_contents=contents,
                    )
                )
            fcntl.flock(root_descriptor, fcntl.LOCK_UN)
        verified_tuple = tuple(
            sorted(
                verified,
                key=lambda item: item.artifact.starting_artifact_content_id,
            )
        )
        policy = validation_settings.policy
        receipt = ExpertSourceReplayContextMaterializationReceipt.mint(
            task_context_binding_id=task_context_binding.task_context_binding_id,
            input_contract_fingerprint=task_context_binding.input_contract_fingerprint,
            target_contract_fingerprint=(
                task_context_binding.target_contract_fingerprint
            ),
            starting_artifacts=tuple(item.artifact for item in verified_tuple),
            materializer_id=policy.source_replay_context_materializer_id,
            materializer_version=policy.source_replay_context_materializer_version,
        )
        self._require_materialization_deadline(deadline)
        return VerifiedSourceReplayContext(
            receipt=receipt,
            starting_artifacts=verified_tuple,
        )

    def _write_object_payloads(
        self,
        payloads: Mapping[str, bytes],
        object_payload_descriptor: int,
    ) -> None:
        for payload in payloads.values():
            if (
                not isinstance(payload, bytes)
                or len(payload) > self.sanitation_settings.max_file_bytes
            ):
                raise RunBundlePublicationError(
                    "content-addressed object exceeds its configured bound"
                )
            object_name = tree_or_blob_digest(payload)[7:]
            if os.access(
                object_name,
                os.F_OK,
                dir_fd=object_payload_descriptor,
                follow_symlinks=False,
            ):
                if (
                    _read_regular_file_at(
                        object_payload_descriptor,
                        object_name,
                        maximum_bytes=self.sanitation_settings.max_file_bytes,
                        required_mode=_IMMUTABLE_FILE_MODE,
                    )
                    != payload
                ):
                    raise RunBundlePublicationError(
                        "content-addressed object collision"
                    )
                _discard_staged_regular_file_at(
                    object_payload_descriptor,
                    f".{object_name}.tmp",
                    required_mode=_IMMUTABLE_FILE_MODE,
                    maximum_bytes=self.sanitation_settings.max_file_bytes,
                )
                continue
            _write_atomic_file_at(
                object_payload_descriptor,
                object_name,
                payload,
                mode=_IMMUTABLE_FILE_MODE,
                maximum_staging_bytes=self.sanitation_settings.max_file_bytes,
            )

    def _commit_manifest_directory(
        self,
        *,
        parent_descriptor: int,
        content_key: str,
        staging_prefix: str,
        payload: bytes,
    ) -> None:
        staging_name = f".{staging_prefix}.{content_key}.tmp"
        if os.access(
            content_key,
            os.F_OK,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        ):
            with ExitStack() as descriptors:
                content_descriptor = _open_child_directory(
                    parent_descriptor,
                    content_key,
                    descriptors,
                    mode=_IMMUTABLE_DIRECTORY_MODE,
                    create=False,
                )
                observed = _read_regular_file_at(
                    content_descriptor,
                    BUNDLE_MANIFEST_FILENAME,
                    maximum_bytes=self.sanitation_settings.max_file_bytes,
                    required_mode=_IMMUTABLE_FILE_MODE,
                )
            if observed != payload:
                raise RunBundlePublicationError("content-addressed manifest collision")
            self._discard_manifest_staging(parent_descriptor, staging_name)
            return
        self._discard_manifest_staging(parent_descriptor, staging_name)
        os.mkdir(
            staging_name,
            mode=_STORE_DIRECTORY_MODE,
            dir_fd=parent_descriptor,
        )
        os.fsync(parent_descriptor)
        with ExitStack() as descriptors:
            staging_descriptor = _open_child_directory(
                parent_descriptor,
                staging_name,
                descriptors,
                mode=_STORE_DIRECTORY_MODE,
                create=False,
            )
            _write_new_file_at(
                staging_descriptor,
                BUNDLE_MANIFEST_FILENAME,
                payload,
                mode=_IMMUTABLE_FILE_MODE,
            )
            os.fchmod(staging_descriptor, _IMMUTABLE_DIRECTORY_MODE)
            os.fsync(staging_descriptor)
        os.rename(
            staging_name,
            content_key,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        os.fsync(parent_descriptor)

    def _discard_manifest_staging(
        self,
        parent_descriptor: int,
        staging_name: str,
    ) -> None:
        if not os.access(
            staging_name,
            os.F_OK,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        ):
            return
        metadata = os.stat(
            staging_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        mode = stat.S_IMODE(metadata.st_mode)
        if not stat.S_ISDIR(metadata.st_mode) or mode not in {
            _STORE_DIRECTORY_MODE,
            _IMMUTABLE_DIRECTORY_MODE,
        }:
            raise RunBundlePublicationError(
                "content-addressed manifest staging identity is invalid"
            )
        with ExitStack() as descriptors:
            staging_descriptor = _open_child_directory(
                parent_descriptor,
                staging_name,
                descriptors,
                mode=mode,
                create=False,
            )
            names = _directory_names(
                staging_descriptor,
                2,
                "content-addressed manifest staging closure",
            )
            if names not in {(), (BUNDLE_MANIFEST_FILENAME,)}:
                raise RunBundlePublicationError(
                    "content-addressed manifest staging closure is invalid"
                )
            if names:
                _read_regular_file_at(
                    staging_descriptor,
                    BUNDLE_MANIFEST_FILENAME,
                    maximum_bytes=self.sanitation_settings.max_file_bytes,
                    required_mode=_IMMUTABLE_FILE_MODE,
                )
            if mode == _IMMUTABLE_DIRECTORY_MODE:
                os.fchmod(staging_descriptor, _STORE_DIRECTORY_MODE)
                os.fsync(staging_descriptor)
            if names:
                os.unlink(BUNDLE_MANIFEST_FILENAME, dir_fd=staging_descriptor)
            os.fsync(staging_descriptor)
        os.rmdir(staging_name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)

    def _read_starting_artifact(
        self,
        starting_artifacts_descriptor: int,
        artifact_id: str,
    ) -> ExpertSourceReplayStartingArtifact:
        key = _bundle_key(artifact_id)
        with ExitStack() as descriptors:
            artifact_descriptor = _open_child_directory(
                starting_artifacts_descriptor,
                key,
                descriptors,
                mode=_IMMUTABLE_DIRECTORY_MODE,
                create=False,
            )
            if _directory_names(
                artifact_descriptor,
                2,
                "starting-artifact manifest closure",
            ) != (BUNDLE_MANIFEST_FILENAME,):
                raise RunBundlePublicationError(
                    "starting-artifact manifest closure is not exact"
                )
            payload = _read_regular_file_at(
                artifact_descriptor,
                BUNDLE_MANIFEST_FILENAME,
                maximum_bytes=self.sanitation_settings.max_file_bytes,
                required_mode=_IMMUTABLE_FILE_MODE,
            )
        artifact = ExpertSourceReplayStartingArtifact.from_json_bytes(payload)
        if artifact.starting_artifact_content_id != artifact_id:
            raise RunBundlePublicationError(
                "starting-artifact manifest identity changed"
            )
        return artifact

    @staticmethod
    def _require_materialization_deadline(deadline: float) -> None:
        if time.monotonic() >= deadline:
            raise RunBundlePublicationError(
                "starting-artifact materialization deadline expired"
            )


class StoredSourceReplayContextProvider:
    """Bind the shared CAS to one exact validation materialization policy."""

    def __init__(
        self,
        store: RunBundleStore,
        validation_settings: ExpertValidationSettings,
    ) -> None:
        if (
            type(store) is not RunBundleStore
            or type(validation_settings) is not ExpertValidationSettings
        ):
            raise RunBundlePublicationError(
                "stored replay context requires exact store and validation settings"
            )
        self.store = store
        self.validation_settings = validation_settings

    def materialize_exact(
        self,
        task_context_binding: TaskContextBinding,
        expected_artifact_content_ids: Mapping[str, str],
        limits: TaskEvaluationMaterializationLimits,
    ) -> VerifiedSourceReplayContext:
        return self.store.materialize_exact(
            task_context_binding,
            expected_artifact_content_ids,
            limits,
            validation_settings=self.validation_settings,
        )


class RunBundlePublisher:
    """Publish one sanitized capture without GitHub calls or interpretation."""

    def __init__(
        self,
        state_root: str | Path,
        settings: CaptureSettings,
        sanitation_settings: SanitationSettings,
    ):
        self.settings = settings
        self.sanitation_settings = sanitation_settings
        self.root = Path(os.path.abspath(state_root))
        state_path = Path(settings.state_path)
        quarantine_path = Path(settings.quarantine_path)
        if state_path.is_absolute() or quarantine_path.is_absolute():
            raise RunBundlePublicationError(
                "capture storage paths must be workspace relative"
            )
        if self.root.parts[-len(state_path.parts) :] != state_path.parts:
            raise RunBundlePublicationError(
                "bundle store is outside the configured workspace state path"
            )
        self.workspace_root = Path(*self.root.parts[: -len(state_path.parts)])
        if self.workspace_root in {Path("/"), Path.home()}:
            raise RunBundlePublicationError("bundle workspace root is unsafe")
        self.state_root = self.root
        self.sanitized_root = self.state_root / "sanitized"
        self.quarantine_root = self.workspace_root / quarantine_path
        self.store = RunBundleStore.initialize(
            self.root,
            self.settings,
            self.sanitation_settings,
        )
        self.runs_identity = self.store.identity.runs

    def publish(
        self,
        capture: ValidatedCapture,
        sanitized: SanitizedCapture,
    ) -> StoredRunBundle:
        self._validate_publication_paths(capture, sanitized)
        self._validate_sanitized_capture(capture, sanitized)
        with ExitStack() as descriptors:
            root_descriptor = _open_absolute_directory(
                self.root,
                descriptors,
                required_mode=_STORE_DIRECTORY_MODE,
            )
            _require_descriptor_identity(
                root_descriptor,
                self.store.identity.root,
            )
            fcntl.flock(root_descriptor, fcntl.LOCK_EX)
            (
                object_payload_descriptor,
                bundles_descriptor,
                runs_descriptor,
            ) = self._open_publication_layout(root_descriptor, descriptors)
            run_key = _run_key(capture.manifest.run_id)
            run_descriptor = _open_child_directory(
                runs_descriptor,
                run_key,
                descriptors,
                mode=_STORE_DIRECTORY_MODE,
                create=True,
            )
            _require_child_directory_binding(
                runs_descriptor,
                run_key,
                run_descriptor,
            )
            current = self._load_current(capture, run_descriptor)
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
                    stored = current
                else:
                    stored = self._publish_new_bundle(
                        capture,
                        sanitized,
                        current,
                        object_payload_descriptor,
                        bundles_descriptor,
                        runs_descriptor,
                        run_key,
                        run_descriptor,
                    )
            else:
                stored = self._publish_new_bundle(
                    capture,
                    sanitized,
                    None,
                    object_payload_descriptor,
                    bundles_descriptor,
                    runs_descriptor,
                    run_key,
                    run_descriptor,
                )
            _require_child_directory_binding(
                root_descriptor,
                "runs",
                runs_descriptor,
            )
            _require_child_directory_binding(
                runs_descriptor,
                run_key,
                run_descriptor,
            )
            visible_current = self._load_current(capture, run_descriptor)
            _require_child_directory_binding(
                root_descriptor,
                "runs",
                runs_descriptor,
            )
            _require_child_directory_binding(
                runs_descriptor,
                run_key,
                run_descriptor,
            )
            if (
                visible_current is None
                or visible_current.manifest.bundle_id != stored.manifest.bundle_id
            ):
                raise RunBundlePublicationError(
                    "published bundle is not the visible run frontier"
                )
            _discard_staged_regular_file_at(
                run_descriptor,
                f".{BUNDLE_CURRENT_FILENAME}.tmp",
                required_mode=_MUTABLE_CONTROL_MODE,
                maximum_bytes=self.sanitation_settings.max_file_bytes,
            )
            stored = self.store.require_exact(stored.manifest.bundle_id)
            _require_child_directory_binding(
                root_descriptor,
                "runs",
                runs_descriptor,
            )
            _require_child_directory_binding(
                runs_descriptor,
                run_key,
                run_descriptor,
            )
            self._prune_quarantine(capture)
            fcntl.flock(root_descriptor, fcntl.LOCK_UN)
        self._remove_sanitized_capture(sanitized)
        return stored

    def _open_publication_layout(
        self,
        root_descriptor: int,
        descriptors: ExitStack,
    ) -> tuple[int, int, int]:
        objects_descriptor = _open_child_directory(
            root_descriptor,
            "objects",
            descriptors,
            mode=_STORE_DIRECTORY_MODE,
            create=False,
        )
        _require_descriptor_identity(
            objects_descriptor,
            self.store.identity.objects,
        )
        object_payload_descriptor = _open_child_directory(
            objects_descriptor,
            "sha256",
            descriptors,
            mode=_STORE_DIRECTORY_MODE,
            create=False,
        )
        _require_descriptor_identity(
            object_payload_descriptor,
            self.store.identity.object_payloads,
        )
        bundles_descriptor = _open_child_directory(
            root_descriptor,
            "bundles",
            descriptors,
            mode=_STORE_DIRECTORY_MODE,
            create=False,
        )
        _require_descriptor_identity(
            bundles_descriptor,
            self.store.identity.bundles,
        )
        runs_descriptor = _open_child_directory(
            root_descriptor,
            "runs",
            descriptors,
            mode=_STORE_DIRECTORY_MODE,
            create=False,
        )
        _require_descriptor_identity(
            runs_descriptor,
            self.runs_identity,
        )
        return object_payload_descriptor, bundles_descriptor, runs_descriptor

    def _publish_new_bundle(
        self,
        capture: ValidatedCapture,
        sanitized: SanitizedCapture,
        current: StoredRunBundle | None,
        object_payload_descriptor: int,
        bundles_descriptor: int,
        runs_descriptor: int,
        run_key: str,
        run_descriptor: int,
    ) -> StoredRunBundle:
        manifest = self._build_manifest(capture, sanitized, current)
        manifest_payload = manifest.to_json_bytes()
        if (
            len(manifest.checksums) > self.settings.bundle_entry_limit
            or len(manifest_payload) > self.sanitation_settings.max_file_bytes
        ):
            raise RunBundlePublicationError("bundle manifest exceeds configured limits")
        self._write_objects(sanitized, object_payload_descriptor)
        self.store._commit_manifest_directory(
            parent_descriptor=bundles_descriptor,
            content_key=_bundle_key(manifest.bundle_id),
            staging_prefix="bundle",
            payload=manifest_payload,
        )
        stored = self.store.require_exact(manifest.bundle_id)
        marker = _current_marker_payload(manifest)
        observed_current = self._load_current(capture, run_descriptor)
        if (current is None) != (observed_current is None) or (
            current is not None
            and observed_current is not None
            and current.manifest.bundle_id != observed_current.manifest.bundle_id
        ):
            raise RunBundlePublicationError("bundle current changed during publication")
        _require_child_directory_binding(
            runs_descriptor,
            run_key,
            run_descriptor,
        )
        _write_atomic_file_at(
            run_descriptor,
            BUNDLE_CURRENT_FILENAME,
            marker,
            mode=_MUTABLE_CONTROL_MODE,
            maximum_staging_bytes=self.sanitation_settings.max_file_bytes,
        )
        _require_child_directory_binding(
            runs_descriptor,
            run_key,
            run_descriptor,
        )
        return stored

    @staticmethod
    def _build_manifest(
        capture: ValidatedCapture,
        sanitized: SanitizedCapture,
        current: StoredRunBundle | None,
    ) -> RunBundle:
        descriptor = capture.descriptor
        return RunBundle.mint(
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
            capture_descriptor_ref=descriptor.artifact_refs["capture_descriptor"],
            checkpoint_ref=descriptor.artifact_refs["checkpoint"],
            execution_event_journal_ref=descriptor.artifact_refs[
                "execution_event_journal"
            ],
            idea_archive_ref=descriptor.artifact_refs["idea_archive"],
            experiment_history_ref=descriptor.artifact_refs["experiment_history"],
            sanitation_report_ref=SANITATION_REPORT_REF,
            branch_snapshot_refs=descriptor.branch_snapshot_refs,
            run_log_refs=descriptor.run_log_refs,
            checksums=sanitized.checksums,
        )

    def _load_current(
        self,
        capture: ValidatedCapture,
        run_descriptor: int,
    ) -> StoredRunBundle | None:
        if not os.access(
            BUNDLE_CURRENT_FILENAME,
            os.F_OK,
            dir_fd=run_descriptor,
            follow_symlinks=False,
        ):
            return None
        marker_payload = _read_regular_file_at(
            run_descriptor,
            BUNDLE_CURRENT_FILENAME,
            maximum_bytes=self.sanitation_settings.max_file_bytes,
            required_mode=_MUTABLE_CONTROL_MODE,
        )
        marker = _parse_current_marker(marker_payload)
        stored = self.store.require_exact(marker["bundle_id"])
        if stored.manifest.capture_generation != marker["capture_generation"]:
            raise RunBundlePublicationError("bundle current marker generation changed")
        self._require_current_run_identity(capture, stored)
        return stored

    @staticmethod
    def _require_current_run_identity(
        capture: ValidatedCapture,
        stored: StoredRunBundle,
    ) -> None:
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

    def _write_objects(
        self,
        sanitized: SanitizedCapture,
        object_payload_descriptor: int,
    ) -> None:
        total_bytes = 0
        payloads = {}
        for relative_path, digest in sorted(sanitized.checksums.items()):
            payload = read_restricted_regular_file(
                sanitized.path,
                relative_path,
                RunBundlePublicationError,
                maximum_bytes=self.sanitation_settings.max_file_bytes,
            )
            total_bytes += len(payload)
            if total_bytes > self.settings.bundle_asset_size_bytes:
                raise RunBundlePublicationError("sanitized payload byte limit exceeded")
            if tree_or_blob_digest(payload) != digest:
                raise RunBundlePublicationError("sanitized payload digest changed")
            payloads[relative_path] = payload
        self.store._write_object_payloads(payloads, object_payload_descriptor)

    def _validate_sanitized_capture(
        self,
        capture: ValidatedCapture,
        sanitized: SanitizedCapture,
    ) -> None:
        if (
            len(capture.manifest.artifact_refs) > self.settings.bundle_entry_limit
            or len(sanitized.checksums) > self.settings.bundle_entry_limit
        ):
            raise RunBundlePublicationError("sanitized closure entry limit exceeded")
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
        entry_count = 0
        total_bytes = 0
        if not sanitized.path.is_dir() or sanitized.path.is_symlink():
            raise RunBundlePublicationError("sanitized root is not a real directory")
        for path in sanitized.path.rglob("*"):
            entry_count += 1
            if entry_count > self.settings.bundle_entry_limit:
                raise RunBundlePublicationError(
                    "sanitized closure entry limit exceeded"
                )
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
                if metadata.st_size > self.sanitation_settings.max_file_bytes:
                    raise RunBundlePublicationError(
                        "sanitized payload exceeds configured size limit"
                    )
                total_bytes += metadata.st_size
                if total_bytes > self.settings.bundle_asset_size_bytes:
                    raise RunBundlePublicationError(
                        "sanitized payload byte limit exceeded"
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
                maximum_bytes=self.sanitation_settings.max_file_bytes,
            )
            if tree_or_blob_digest(payload) != digest:
                raise RunBundlePublicationError("sanitized payload digest changed")
        report_payload = read_restricted_regular_file(
            sanitized.path,
            SANITATION_REPORT_REF,
            RunBundlePublicationError,
            maximum_bytes=self.sanitation_settings.max_file_bytes,
        )
        if SanitationReport.from_json_bytes(report_payload) != sanitized.report:
            raise RunBundlePublicationError("sanitation report bytes changed")

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
        lock_path = run_root / CAPTURE_EXPORT_LOCK_FILENAME
        _reject_symlink_components(lock_path)
        lock_descriptor = os.open(
            lock_path,
            os.O_RDWR | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        with os.fdopen(lock_descriptor, "r+b") as lock_handle:
            lock_metadata = os.fstat(lock_handle.fileno())
            if (
                not stat.S_ISREG(lock_metadata.st_mode)
                or lock_metadata.st_nlink != 1
                or stat.S_IMODE(lock_metadata.st_mode) != _MUTABLE_CONTROL_MODE
                or lock_metadata.st_size != 0
            ):
                raise RunBundlePublicationError(
                    "capture export lock identity is invalid"
                )
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            self._prune_quarantine_locked(capture, run_root)
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def _prune_quarantine_locked(
        self,
        capture: ValidatedCapture,
        run_root: Path,
    ) -> None:
        marker_payload = read_restricted_regular_file(
            run_root,
            CAPTURE_CURRENT_FILENAME,
            RunBundlePublicationError,
            maximum_bytes=self.sanitation_settings.max_file_bytes,
        )
        marker = parse_json_bytes(marker_payload)
        if (
            not isinstance(marker, dict)
            or set(marker) != {"capture_manifest_id", "generation", "path"}
            or not isinstance(marker["capture_manifest_id"], str)
            or type(marker["generation"]) is not int
            or marker["generation"] < capture.manifest.capture_generation
            or marker["path"] != f"generation-{marker['generation']:020d}"
            or marker_payload != canonical_json_bytes(marker) + b"\n"
        ):
            raise RunBundlePublicationError(
                "capture current marker is invalid for retention"
            )
        require_content_id(marker["capture_manifest_id"], "capture current ID")
        current_manifest_payload = read_restricted_regular_file(
            run_root,
            f"{marker['path']}/{CAPTURE_MANIFEST_FILENAME}",
            RunBundlePublicationError,
            maximum_bytes=self.sanitation_settings.max_file_bytes,
        )
        current_manifest = CaptureManifest.from_json_bytes(current_manifest_payload)
        if (
            current_manifest.to_json_bytes() != current_manifest_payload
            or current_manifest.capture_manifest_id != marker["capture_manifest_id"]
            or current_manifest.capture_generation != marker["generation"]
            or (
                current_manifest.scope_contract_id,
                current_manifest.scope_id,
                current_manifest.run_id,
                current_manifest.campaign_id,
                current_manifest.configuration_fingerprint,
            )
            != (
                capture.manifest.scope_contract_id,
                capture.manifest.scope_id,
                capture.manifest.run_id,
                capture.manifest.campaign_id,
                capture.manifest.configuration_fingerprint,
            )
        ):
            raise RunBundlePublicationError(
                "capture current manifest is invalid for retention"
            )
        committed_generations: list[tuple[Path, tuple[int, int]]] = []
        entry_count = 0
        for path in run_root.iterdir():
            entry_count += 1
            if entry_count > self.settings.bundle_entry_limit:
                raise RunBundlePublicationError(
                    "capture retention index exceeds configured entry limit"
                )
            if path.is_symlink():
                raise RunBundlePublicationError("capture retention contains a symlink")
            if not path.name.startswith("generation-"):
                continue
            if (
                not path.is_dir()
                or re.fullmatch(r"generation-[0-9]{20}", path.name) is None
            ):
                raise RunBundlePublicationError(
                    "capture retention contains an invalid generation"
                )
            generation = int(path.name.removeprefix("generation-"))
            if generation <= marker["generation"]:
                committed_generations.append(
                    (
                        path,
                        restricted_directory_identity(
                            run_root,
                            path.name,
                            RunBundlePublicationError,
                        ),
                    )
                )
        committed_generations.sort(key=lambda item: item[0].name)
        if not any(path.name == marker["path"] for path, _ in committed_generations):
            raise RunBundlePublicationError(
                "capture current generation is absent during retention"
            )
        removable = committed_generations[
            : -self.settings.quarantine_retention_generations
        ]
        for path, identity in removable:
            remove_restricted_directory(
                run_root,
                path.name,
                identity,
                RunBundlePublicationError,
            )
