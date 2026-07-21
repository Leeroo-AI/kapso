"""Deterministic secret, path, artifact-class, and license sanitation gate."""

from __future__ import annotations

import fcntl
import os
import re
import stat
import uuid
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Mapping

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    parse_json_bytes,
    tree_or_blob_digest,
)
from kapso.cross_run.capture.safety import (
    read_restricted_regular_file,
    remove_restricted_directory,
    restricted_directory_identity,
)
from kapso.cross_run.capture.validator import ValidatedCapture
from kapso.cross_run.record_contracts import (
    ExecutionRevisionEvent,
    SANITATION_REPORT_SCHEMA,
    SanitationReport,
)
from kapso.cross_run.sanitation_rules import (
    SANITATION_SCANNER_VERSION,
    sanitation_policy_fingerprint,
    scan_text_artifact,
)
from kapso.cross_run.settings import CaptureSettings, SanitationSettings
from kapso.execution.memories.experiment_memory.store import EXPERIMENT_HISTORY_SCHEMA

SANITATION_REPORT_REF = "sanitation_report.json"
CAPTURE_CURRENT_FILENAME = "current.json"
_REDACTED_RECORD_FIELDS = {
    "error_message": "",
    "evaluation_integrity_error": "",
    "external_evaluation_error": "",
    "external_evaluation_metadata": {},
    "feedback": "",
    "technical_difficulties": "",
}
_REDACTED_NODE_FIELDS = {
    "agent_output": "",
    "code_changes_summary": "",
    "code_diff": "",
    "error_message": "",
    "evaluation_integrity_error": "",
    "evaluation_output": "",
    "evaluation_script_path": "",
    "external_evaluation_error": "",
    "external_evaluation_metadata": {},
    "feedback": "",
    "technical_difficulties": "",
    "workspace_dir": "",
}


class SanitationRejectedError(ValueError):
    """Deterministic policy rejected a capture from durable publication."""

    def __init__(self, report: "SanitationReport", report_path: Path):
        super().__init__(f"capture rejected by sanitation policy: {report.report_id}")
        self.report = report
        self.report_path = report_path


@dataclass(frozen=True)
class SanitizedCapture:
    path: Path
    directory_identity: tuple[int, int]
    report: SanitationReport
    artifact_refs: Mapping[str, str]
    checksums: Mapping[str, str]


def _open_or_create_output_root(
    path: Path,
    descriptors: ExitStack,
) -> tuple[Path, int, tuple[tuple[int, str, int], ...]]:
    absolute = Path(os.path.abspath(path))
    if absolute == Path(absolute.anchor):
        raise ValueError("sanitized output root cannot be a filesystem root")
    descriptor = os.open(
        absolute.anchor,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    descriptors.callback(os.close, descriptor)
    parent_descriptor = descriptor
    placements = []
    for name in absolute.parts[1:]:
        parent_descriptor = descriptor
        if not os.access(
            name,
            os.F_OK,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        ):
            os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
        descriptors.callback(os.close, descriptor)
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("sanitized output path is not a real directory")
        placements.append((parent_descriptor, name, descriptor))
    os.fchmod(descriptor, 0o700)
    _require_directory_chain(tuple(placements))
    return absolute, descriptor, tuple(placements)


def _open_or_create_child_directory(
    parent_descriptor: int,
    name: str,
    descriptors: ExitStack,
) -> int:
    if not os.access(
        name,
        os.F_OK,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    ):
        os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        dir_fd=parent_descriptor,
    )
    descriptors.callback(os.close, descriptor)
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("sanitized output child is not a real directory")
    os.fchmod(descriptor, 0o700)
    _require_same_directory_entry(parent_descriptor, name, descriptor)
    return descriptor


def _create_child_directory(
    parent_descriptor: int,
    name: str,
    descriptors: ExitStack,
) -> int:
    os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        dir_fd=parent_descriptor,
    )
    descriptors.callback(os.close, descriptor)
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("sanitized staging child is not a real directory")
    os.fchmod(descriptor, 0o700)
    _require_same_directory_entry(parent_descriptor, name, descriptor)
    return descriptor


def _require_same_directory_entry(
    parent_descriptor: int,
    name: str,
    descriptor: int,
) -> None:
    expected = os.fstat(descriptor)
    current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    if not stat.S_ISDIR(current.st_mode) or (current.st_dev, current.st_ino) != (
        expected.st_dev,
        expected.st_ino,
    ):
        raise ValueError("sanitized output directory was replaced")


def _require_directory_chain(
    placements: tuple[tuple[int, str, int], ...],
) -> None:
    for parent_descriptor, name, descriptor in placements:
        _require_same_directory_entry(parent_descriptor, name, descriptor)


def _fsync_directory_chain(
    placements: tuple[tuple[int, str, int], ...],
) -> None:
    for parent_descriptor, _, descriptor in reversed(placements):
        os.fsync(descriptor)
        os.fsync(parent_descriptor)


def _open_sanitation_lock(
    root_descriptor: int,
    descriptors: ExitStack,
) -> tuple[BinaryIO, tuple[int, int]]:
    descriptor = os.open(
        ".sanitation.lock",
        os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
        0o600,
        dir_fd=root_descriptor,
    )
    handle = descriptors.enter_context(os.fdopen(descriptor, "r+b"))
    metadata = os.fstat(handle.fileno())
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_mode & 0o077
    ):
        raise ValueError("sanitation lock is not a private regular file")
    identity = (metadata.st_dev, metadata.st_ino)
    _require_same_file_entry(root_descriptor, ".sanitation.lock", identity)
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    _require_same_file_entry(root_descriptor, ".sanitation.lock", identity)
    return handle, identity


def _write_new_file(
    parent_descriptor: int,
    name: str,
    payload: bytes,
) -> tuple[int, int]:
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=parent_descriptor,
    )
    with os.fdopen(descriptor, "wb") as handle:
        metadata = os.fstat(handle.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError("sanitized artifact is not an independent regular file")
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        os.fchmod(handle.fileno(), 0o600)
        metadata = os.fstat(handle.fileno())
    return metadata.st_dev, metadata.st_ino


def _read_existing_file(
    parent_descriptor: int,
    name: str,
) -> tuple[bytes, tuple[int, int]]:
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_NOFOLLOW,
        dir_fd=parent_descriptor,
    )
    with os.fdopen(descriptor, "rb") as handle:
        metadata = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_mode & 0o077
        ):
            raise ValueError("sanitation report is not a private regular file")
        return handle.read(), (metadata.st_dev, metadata.st_ino)


def _require_same_file_entry(
    parent_descriptor: int,
    name: str,
    expected_identity: tuple[int, int],
) -> None:
    current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    if (
        not stat.S_ISREG(current.st_mode)
        or current.st_nlink != 1
        or current.st_mode & 0o077
        or (current.st_dev, current.st_ino) != expected_identity
    ):
        raise ValueError("sanitized output file was replaced")


class SanitationGate:
    """Admit an exact safe byte closure or emit an immutable rejection report."""

    def __init__(
        self,
        capture_settings: CaptureSettings,
        settings: SanitationSettings,
    ):
        self.capture_settings = capture_settings
        self.settings = settings

    def sanitize(
        self,
        capture: ValidatedCapture,
        output_root: str | Path,
    ) -> SanitizedCapture:
        findings: list[dict[str, str]] = []
        exclusions: list[dict[str, str]] = []
        taints: set[str] = set()
        admitted: dict[str, str] = {}
        admitted_refs = set(capture.manifest.artifact_refs.values())
        raw_payloads: dict[str, bytes] = {}
        for relative_path in sorted(admitted_refs):
            payload = read_restricted_regular_file(
                capture.path,
                relative_path,
                ValueError,
            )
            if (
                tree_or_blob_digest(payload)
                != capture.manifest.checksums[relative_path]
            ):
                raise ValueError("capture artifact changed before sanitation")
            raw_payloads[relative_path] = payload
        projected_payloads = self._safe_authority_projections(capture)

        for branch in capture.branch_snapshots:
            for exclusion in branch.excluded_files:
                exclusions.append(
                    {"path": exclusion["path"], "reason": exclusion["reason"]}
                )
                taints.add(f"excluded_{exclusion['reason']}")

        for relative_path in sorted(admitted_refs):
            payload = projected_payloads.get(relative_path, raw_payloads[relative_path])
            for finding in scan_text_artifact(
                self.settings,
                relative_path,
                payload,
            ):
                findings.append(
                    {
                        "code": finding.code,
                        "evidence_digest": finding.evidence_digest,
                        "path": finding.relative_path,
                        "severity": finding.severity,
                    }
                )
                if finding.code == "unclassified_license":
                    taints.add("unclassified_license")
            admitted[relative_path] = tree_or_blob_digest(payload)

        sorted_findings = tuple(
            {
                "code": code,
                "evidence_digest": evidence_digest,
                "path": path,
                "severity": severity,
            }
            for path, code, evidence_digest, severity in sorted(
                {
                    (
                        item["path"],
                        item["code"],
                        item["evidence_digest"],
                        item["severity"],
                    )
                    for item in findings
                }
            )
        )
        sorted_exclusions = tuple(
            {"path": path, "reason": reason}
            for path, reason in sorted(
                {(item["path"], item["reason"]) for item in exclusions}
            )
        )
        rejected = any(item["severity"] == "reject" for item in sorted_findings)
        report = SanitationReport.mint(
            schema=SANITATION_REPORT_SCHEMA,
            capture_manifest_id=capture.manifest.capture_manifest_id,
            scope_id=capture.manifest.scope_id,
            task_family_id=capture.descriptor.task_context_binding.task_family_id,
            policy_version=self.settings.policy_version,
            policy_fingerprint=sanitation_policy_fingerprint(self.settings),
            scanner_version=SANITATION_SCANNER_VERSION,
            status="rejected" if rejected else "admitted",
            findings=sorted_findings,
            excluded_paths=sorted_exclusions,
            taint_sources=tuple(sorted(taints)),
            admitted_refs={} if rejected else admitted,
        )
        report_payload = report.to_json_bytes()
        if rejected:
            report_path = self._persist_rejection(
                Path(output_root),
                report,
                report_payload,
            )
            self._prune_after_rejection(capture, Path(output_root))
            raise SanitationRejectedError(report, report_path)

        admitted_payloads = {
            relative_path: projected_payloads.get(
                relative_path,
                raw_payloads[relative_path],
            )
            for relative_path in sorted(admitted)
        }
        admitted_payloads[SANITATION_REPORT_REF] = report_payload
        staging, staging_identity = self._persist_admitted(
            Path(output_root),
            report,
            admitted_payloads,
        )
        checksums = dict(admitted)
        checksums[SANITATION_REPORT_REF] = tree_or_blob_digest(report_payload)
        return SanitizedCapture(
            path=staging,
            directory_identity=staging_identity,
            report=report,
            artifact_refs=dict(capture.manifest.artifact_refs),
            checksums=checksums,
        )

    def _prune_after_rejection(
        self,
        capture: ValidatedCapture,
        output_root: Path,
    ) -> None:
        absolute_output_root = Path(os.path.abspath(output_root))
        state_path = Path(self.capture_settings.state_path)
        quarantine_path = Path(self.capture_settings.quarantine_path)
        expected_output_tail = (*state_path.parts, "sanitized")
        if absolute_output_root.parts[-len(expected_output_tail) :] != (
            expected_output_tail
        ):
            raise ValueError("sanitized output root is outside configured state")
        workspace = Path(*absolute_output_root.parts[: -len(expected_output_tail)])
        generation_name = f"generation-{capture.manifest.capture_generation:020d}"
        run_key = tree_or_blob_digest(capture.manifest.run_id.encode("utf-8"))[7:]
        run_root = workspace / quarantine_path / "runs" / run_key
        expected_capture_path = run_root / generation_name
        if Path(os.path.abspath(capture.path)) != expected_capture_path:
            raise ValueError("rejected capture path is outside configured quarantine")
        marker_payload = read_restricted_regular_file(
            run_root,
            CAPTURE_CURRENT_FILENAME,
            ValueError,
        )
        marker = parse_json_bytes(marker_payload)
        if marker != {
            "capture_manifest_id": capture.manifest.capture_manifest_id,
            "generation": capture.manifest.capture_generation,
            "path": generation_name,
        }:
            raise ValueError("rejected capture is not the current generation")
        generations = []
        for path in run_root.iterdir():
            if path.is_symlink():
                raise ValueError("quarantine run root contains a symlink")
            if not path.is_dir():
                continue
            if re.fullmatch(r"generation-[0-9]{20}", path.name) is None:
                raise ValueError("quarantine run root contains an invalid generation")
            generations.append(
                (
                    path,
                    restricted_directory_identity(
                        run_root,
                        path.name,
                        ValueError,
                    ),
                )
            )
        generations.sort(key=lambda item: item[0].name)
        if (
            not generations
            or generations[-1][0].name != generation_name
            or generations[-1][1] != capture.directory_identity
        ):
            raise ValueError("rejected capture generation identity changed")
        removable = generations[
            : -self.capture_settings.quarantine_retention_generations
        ]
        for path, identity in removable:
            remove_restricted_directory(
                run_root,
                path.name,
                identity,
                ValueError,
            )

    @staticmethod
    def _persist_rejection(
        output_root: Path,
        report: SanitationReport,
        report_payload: bytes,
    ) -> Path:
        report_name = f"{report.report_id.rsplit(':', 1)[-1]}.json"
        with ExitStack() as descriptors:
            root, root_descriptor, root_placements = _open_or_create_output_root(
                output_root, descriptors
            )
            _, lock_identity = _open_sanitation_lock(root_descriptor, descriptors)
            rejection_descriptor = _open_or_create_child_directory(
                root_descriptor,
                "rejections",
                descriptors,
            )
            if os.access(
                report_name,
                os.F_OK,
                dir_fd=rejection_descriptor,
                follow_symlinks=False,
            ):
                existing_payload, report_identity = _read_existing_file(
                    rejection_descriptor,
                    report_name,
                )
                if existing_payload != report_payload:
                    raise ValueError("sanitation report identity has conflicting bytes")
            else:
                report_identity = _write_new_file(
                    rejection_descriptor,
                    report_name,
                    report_payload,
                )
            _require_same_file_entry(
                rejection_descriptor,
                report_name,
                report_identity,
            )
            os.fsync(rejection_descriptor)
            _fsync_directory_chain(root_placements)
            _require_same_directory_entry(
                root_descriptor,
                "rejections",
                rejection_descriptor,
            )
            _require_same_file_entry(
                root_descriptor,
                ".sanitation.lock",
                lock_identity,
            )
            _require_directory_chain(root_placements)
        return root / "rejections" / report_name

    @staticmethod
    def _persist_admitted(
        output_root: Path,
        report: SanitationReport,
        payloads: Mapping[str, bytes],
    ) -> tuple[Path, tuple[int, int]]:
        staging_name = (
            f".sanitized.{report.report_id.rsplit(':', 1)[-1]}.{uuid.uuid4().hex}"
        )
        with ExitStack() as descriptors:
            root, root_descriptor, root_placements = _open_or_create_output_root(
                output_root, descriptors
            )
            _, lock_identity = _open_sanitation_lock(root_descriptor, descriptors)
            staging_descriptor = _create_child_directory(
                root_descriptor,
                staging_name,
                descriptors,
            )
            staging_metadata = os.fstat(staging_descriptor)
            staging_identity = (
                staging_metadata.st_dev,
                staging_metadata.st_ino,
            )
            directory_descriptors: dict[PurePosixPath, int] = {
                PurePosixPath("."): staging_descriptor
            }
            directory_identities: dict[PurePosixPath, tuple[int, int]] = {}
            expected_children: dict[PurePosixPath, set[str]] = {
                PurePosixPath("."): set()
            }
            paths = tuple(PurePosixPath(path) for path in payloads)
            directory_paths = {
                parent
                for path in paths
                for parent in path.parents
                if parent != PurePosixPath(".")
            }
            for path in sorted(
                directory_paths,
                key=lambda item: (len(item.parts), item.as_posix()),
            ):
                parent = path.parent
                descriptor = _create_child_directory(
                    directory_descriptors[parent],
                    path.name,
                    descriptors,
                )
                directory_descriptors[path] = descriptor
                metadata = os.fstat(descriptor)
                directory_identities[path] = (metadata.st_dev, metadata.st_ino)
                expected_children.setdefault(parent, set()).add(path.name)
                expected_children[path] = set()

            file_identities: dict[PurePosixPath, tuple[int, int]] = {}
            for path in sorted(paths, key=lambda item: item.as_posix()):
                file_identities[path] = _write_new_file(
                    directory_descriptors[path.parent],
                    path.name,
                    payloads[path.as_posix()],
                )
                expected_children[path.parent].add(path.name)

            for path, descriptor in directory_descriptors.items():
                if set(os.listdir(descriptor)) != expected_children[path]:
                    raise ValueError("sanitized output file closure changed")
            for path, identity in file_identities.items():
                _require_same_file_entry(
                    directory_descriptors[path.parent],
                    path.name,
                    identity,
                )
            for path, identity in directory_identities.items():
                current = os.stat(
                    path.name,
                    dir_fd=directory_descriptors[path.parent],
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISDIR(current.st_mode)
                    or (current.st_dev, current.st_ino) != identity
                ):
                    raise ValueError("sanitized output directory was replaced")
            for path in sorted(
                directory_descriptors,
                key=lambda item: len(item.parts),
                reverse=True,
            ):
                os.fsync(directory_descriptors[path])
            _fsync_directory_chain(root_placements)
            _require_same_directory_entry(
                root_descriptor,
                staging_name,
                staging_descriptor,
            )
            _require_same_file_entry(
                root_descriptor,
                ".sanitation.lock",
                lock_identity,
            )
            _require_directory_chain(root_placements)
        return root / staging_name, staging_identity

    @staticmethod
    def _safe_measurement_fields(
        projection: dict[str, Any],
        allowed_metric_names: frozenset[str],
    ) -> None:
        projection["metrics"] = {
            name: value
            for name, value in projection["metrics"].items()
            if name in allowed_metric_names
        }
        if projection["primary_metric"] not in allowed_metric_names:
            projection["primary_metric"] = None
        projection["phase_telemetry"] = {}
        projected_attempts = []
        for attempt in projection["evaluation_attempts"]:
            projected_attempt = dict(attempt)
            projected_attempt["metrics"] = {
                name: value
                for name, value in attempt["metrics"].items()
                if name in allowed_metric_names
            }
            projected_attempts.append(projected_attempt)
        projection["evaluation_attempts"] = projected_attempts

    @classmethod
    def _safe_record_projection(
        cls,
        record: Mapping[str, Any],
        allowed_metric_names: frozenset[str],
    ) -> dict[str, Any]:
        projected = dict(record)
        for field, safe_value in _REDACTED_RECORD_FIELDS.items():
            projected[field] = safe_value
        cls._safe_measurement_fields(projected, allowed_metric_names)
        return projected

    @staticmethod
    def _safe_archive_projection(archive: Mapping[str, Any]) -> dict[str, Any]:
        projected = dict(archive)
        projected_batches = []
        for batch in archive["batches"]:
            projected_batch = dict(batch)
            projected_evidence = dict(batch["evidence_snapshot"])
            projected_experiments = []
            for experiment in projected_evidence["experiments"]:
                projected_experiment = dict(experiment)
                projected_experiment["feedback"] = ""
                projected_experiment["technical_difficulty"] = None
                projected_experiments.append(projected_experiment)
            projected_evidence["experiments"] = projected_experiments
            projected_batch["evidence_snapshot"] = projected_evidence
            projected_calls = []
            for call in batch["generation_calls"]:
                projected_call = dict(call)
                projected_call["output"] = ""
                projected_call["artifacts"] = []
                projected_calls.append(projected_call)
            projected_batch["generation_calls"] = projected_calls
            if batch["selection_call"] is not None:
                projected_selection_call = dict(batch["selection_call"])
                projected_selection_call["output"] = ""
                projected_selection_call["artifacts"] = []
                projected_batch["selection_call"] = projected_selection_call
            projected_batches.append(projected_batch)
        projected["batches"] = projected_batches
        projected_ideas = []
        for idea in archive["ideas"]:
            projected_idea = dict(idea)
            projected_idea["generation_artifacts"] = []
            projected_ideas.append(projected_idea)
        projected["ideas"] = projected_ideas
        return projected

    def _safe_authority_projections(
        self, capture: ValidatedCapture
    ) -> dict[str, bytes]:
        checkpoint = capture.checkpoint.to_dict()
        checkpoint["current_feedback"] = None
        strategy_state = dict(checkpoint["strategy_state"])
        allowed_metric_names = frozenset(
            fingerprint.metric_name
            for fingerprint in capture.descriptor.evaluation_fingerprints
        )
        projected_archive = self._safe_archive_projection(capture.archive.to_dict())
        projected_nodes = []
        for node in strategy_state["node_history"]:
            projected = dict(node)
            for field, safe_value in _REDACTED_NODE_FIELDS.items():
                projected[field] = safe_value
            self._safe_measurement_fields(projected, allowed_metric_names)
            projected_nodes.append(projected)
        strategy_state["node_history"] = projected_nodes
        strategy_state["previous_errors"] = []
        evaluation_integrity = dict(strategy_state["evaluation_integrity"])
        evaluation_integrity["manifest"] = {}
        evaluation_integrity["fingerprint"] = None
        strategy_state["evaluation_integrity"] = evaluation_integrity
        strategy_state["idea_archive_snapshot"] = projected_archive
        checkpoint["strategy_state"] = strategy_state

        records = tuple(
            self._safe_record_projection(record.to_dict(), allowed_metric_names)
            for record in capture.history.experiments
        )
        history = {
            "schema": EXPERIMENT_HISTORY_SCHEMA,
            "run_id": capture.history.run_id,
            "campaign_id": capture.history.campaign_id,
            "revision": capture.history.revision,
            "objective_direction": capture.history.objective_direction,
            "require_idea_links": capture.history.require_idea_links,
            "records": records,
        }
        journal_payloads = []
        allowed_measurement_names = allowed_metric_names | {"raw_score"}
        for event in capture.events:
            values = event.to_dict()
            values.pop("event_id")
            values["feedback"] = ""
            values["technical_difficulties"] = ""
            values["measurements"] = {
                name: value
                for name, value in values["measurements"].items()
                if name in allowed_measurement_names
            }
            values["projection"] = self._safe_record_projection(
                values["projection"], allowed_metric_names
            )
            sanitized_event = ExecutionRevisionEvent.mint(**values)
            journal_payloads.append(sanitized_event.to_json_bytes() + b"\n")
        refs = capture.descriptor.artifact_refs
        return {
            refs["checkpoint"]: canonical_json_bytes(checkpoint),
            refs["experiment_history"]: canonical_json_bytes(history),
            refs["execution_event_journal"]: b"".join(journal_payloads),
            refs["idea_archive"]: canonical_json_bytes(projected_archive),
        }
