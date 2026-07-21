"""Private, create-only expert candidate package store."""

from __future__ import annotations

import ctypes
import errno
import fcntl
import os
import re
import stat
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import ClassVar, Mapping

from kapso.cross_run.agent_artifacts import (
    CodingAgentWorkspaceAccess,
    coding_agent_artifact_filenames,
)
from kapso.cross_run.canonical import (
    canonical_json_bytes,
    parse_json_bytes,
    require_content_id,
    tree_or_blob_digest,
)
from kapso.cross_run.capture.safety import (
    remove_restricted_directory,
    restricted_directory_identity,
)
from kapso.cross_run.contracts import (
    CodingAgentWorkspaceDelta,
    ExpertCandidateManifest,
    ExpertCandidateOperationRecord,
    ExpertCandidatePatch,
    ExpertCandidateSanitationReport,
    ExpertModuleContract,
    ExpertRepositoryMap,
    ExpertSourceTreeManifest,
    SourceFileDescriptor,
    StrictContract,
)
from kapso.cross_run.expert.candidates import (
    ExpertCandidateClosure,
    ExpertCandidateValidator,
)
from kapso.cross_run.expert.proposal_contract import ExpertCandidateAncestorInput
from kapso.cross_run.expert.triggers import (
    ExpertEvolutionTriggerDecision,
    ExpertTriggerEvidencePacket,
)

_COMMIT_PATH = "COMMITTED.json"
_MANIFEST_PATH = "candidate.json"
_TRIGGER_PACKET_PATH = "trigger-packet.json"
_TRIGGER_DECISION_PATH = "trigger-decision.json"
_PATCH_PATH = "patch.json"
_SOURCE_TREE_PATH = "source-tree.json"
_PARENT_FILES_PATH = "parent-files.json"
_REPOSITORY_MAP_PATH = "repository-map.json"
_OPERATION_PATH = "operation.json"
_SANITATION_PATH = "sanitation.json"
_ANCESTORS_PATH = "ancestors.json"
_MODULE_ROOT = "module-contracts"
_SOURCE_ROOT = "source"
_AGENT_ARTIFACT_ROOT = "agent-artifacts"
_WORKSPACE_DELTA_PATH = "workspace-delta.json"
_RENAME_NOREPLACE = 1
_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_STAGING_PATTERN = re.compile(r"^\.candidate-[A-Za-z0-9_-]+$")


class ExpertCandidateStoreError(ValueError):
    """Candidate package storage is unsafe, corrupt, or conflicting."""


@dataclass(frozen=True)
class ExpertCandidateCommitRecord(StrictContract):
    commit_record_id: str
    candidate_id: str
    file_checksums: Mapping[str, str]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-candidate-commit"
    IDENTITY_FIELD: ClassVar[str] = "commit_record_id"

    def _validate(self) -> None:
        require_content_id(self.candidate_id, "candidate_id")
        if not self.file_checksums:
            raise ExpertCandidateStoreError("candidate commit has no files")
        for relative_path, digest in self.file_checksums.items():
            path = PurePosixPath(relative_path)
            if (
                not relative_path
                or path.is_absolute()
                or path == PurePosixPath(".")
                or ".." in path.parts
                or path.as_posix() != relative_path
                or relative_path == _COMMIT_PATH
            ):
                raise ExpertCandidateStoreError("candidate commit file path is invalid")
            if _DIGEST_PATTERN.fullmatch(digest) is None:
                raise ExpertCandidateStoreError(
                    "candidate commit file digest is invalid"
                )


@dataclass(frozen=True)
class StoredExpertCandidate:
    root: Path
    closure: ExpertCandidateClosure
    commit_record: ExpertCandidateCommitRecord


class ExpertCandidateStore:
    """Atomically seal and reopen complete candidate closures."""

    def __init__(
        self,
        root: Path,
        state_root: Path,
        validator: ExpertCandidateValidator,
    ):
        self._validate_state_root(state_root)
        if (
            not root.is_absolute()
            or root != Path(os.path.abspath(root))
            or root.parent != state_root
            or root.name in {"", ".", ".."}
        ):
            raise ExpertCandidateStoreError(
                "candidate store must be a direct normalized child of its state root"
            )
        self.root = root
        self.state_root = state_root
        self.validator = validator
        self.object_root = root / "objects"
        self.staging_root = root / "staging"
        initialization_lock = state_root / f".{root.name}.initialization.lock"
        with _CandidateStoreLock(
            initialization_lock,
            exclusive=True,
            create=True,
        ):
            self._prepare_layout()
        with self._exclusive_lock():
            self._recover_staging()

    def persist(self, closure: ExpertCandidateClosure) -> StoredExpertCandidate:
        snapshot = replace(
            closure,
            candidate_contents=dict(closure.candidate_contents),
            operation_artifacts=dict(closure.operation_artifacts),
        )
        self.validator.validate(snapshot)
        package_files = self._package_files(snapshot)
        commit_record = ExpertCandidateCommitRecord.mint(
            candidate_id=snapshot.manifest.candidate_id,
            file_checksums={
                path: tree_or_blob_digest(payload)
                for path, payload in sorted(package_files.items())
            },
        )
        destination = self._candidate_path(snapshot.manifest.candidate_id)
        with self._exclusive_lock():
            self._recover_staging()
            if os.path.lexists(destination):
                stored = self._read_unlocked(snapshot.manifest.candidate_id)
                if stored.closure != snapshot or stored.commit_record != commit_record:
                    raise ExpertCandidateStoreError(
                        "candidate identity conflicts with persisted closure"
                    )
                return stored
            with tempfile.TemporaryDirectory(
                prefix=".candidate-",
                dir=self.staging_root,
            ) as staging_name:
                staging = Path(staging_name)
                staging.chmod(0o700)
                for relative_path, payload in package_files.items():
                    self._write_private_file(staging, relative_path, payload)
                self._write_private_file(
                    staging,
                    _COMMIT_PATH,
                    commit_record.to_json_bytes(),
                )
                self._fsync_tree(staging)
                staged = self._read_package(
                    staging,
                    snapshot.manifest.candidate_id,
                )
                if staged.closure != snapshot or staged.commit_record != commit_record:
                    raise ExpertCandidateStoreError(
                        "staged candidate differs from its validated snapshot"
                    )
                self._rename_directory_no_replace(staging, destination)
                self._fsync_directory(self.staging_root)
                self._fsync_directory(self.object_root)
            return self._read_unlocked(snapshot.manifest.candidate_id)

    def read(self, candidate_id: str) -> StoredExpertCandidate:
        require_content_id(candidate_id, "candidate_id")
        with self._shared_lock():
            return self._read_unlocked(candidate_id)

    def _read_unlocked(self, candidate_id: str) -> StoredExpertCandidate:
        return self._read_package(self._candidate_path(candidate_id), candidate_id)

    def _read_package(
        self,
        candidate_root: Path,
        candidate_id: str,
    ) -> StoredExpertCandidate:
        files = self._read_private_tree(candidate_root)
        commit_payload = files.get(_COMMIT_PATH)
        if commit_payload is None:
            raise ExpertCandidateStoreError("candidate package is not committed")
        commit = ExpertCandidateCommitRecord.from_json_bytes(commit_payload)
        if commit_payload != commit.to_json_bytes():
            raise ExpertCandidateStoreError("candidate commit record is not canonical")
        payloads = {
            path: payload for path, payload in files.items() if path != _COMMIT_PATH
        }
        if commit.candidate_id != candidate_id or set(payloads) != set(
            commit.file_checksums
        ):
            raise ExpertCandidateStoreError(
                "candidate package differs from its commit closure"
            )
        for path, payload in payloads.items():
            if tree_or_blob_digest(payload) != commit.file_checksums[path]:
                raise ExpertCandidateStoreError(
                    f"candidate package checksum differs: {path}"
                )
        closure = self._parse_closure(payloads)
        if closure.manifest.candidate_id != candidate_id:
            raise ExpertCandidateStoreError(
                "candidate directory names another manifest"
            )
        self.validator.validate_persisted(closure)
        return StoredExpertCandidate(
            root=candidate_root,
            closure=closure,
            commit_record=commit,
        )

    @staticmethod
    def _package_files(closure: ExpertCandidateClosure) -> dict[str, bytes]:
        files = {
            _MANIFEST_PATH: closure.manifest.to_json_bytes(),
            _TRIGGER_PACKET_PATH: closure.trigger_packet.to_json_bytes(),
            _TRIGGER_DECISION_PATH: closure.trigger_decision.to_json_bytes(),
            _PATCH_PATH: closure.patch.to_json_bytes(),
            _SOURCE_TREE_PATH: closure.candidate_tree.to_json_bytes(),
            _PARENT_FILES_PATH: ExpertCandidateStore._contract_tuple_bytes(
                closure.parent_files
            ),
            _REPOSITORY_MAP_PATH: closure.repository_map.to_json_bytes(),
            _OPERATION_PATH: closure.operation.to_json_bytes(),
            _WORKSPACE_DELTA_PATH: closure.workspace_delta.to_json_bytes(),
            _SANITATION_PATH: closure.sanitation_report.to_json_bytes(),
            _ANCESTORS_PATH: ExpertCandidateStore._contract_tuple_bytes(
                closure.ancestor_inputs
            ),
        }
        for module in closure.module_contracts:
            digest = module.module_contract_id.rsplit(":", 1)[1]
            files[f"{_MODULE_ROOT}/{digest}.json"] = module.to_json_bytes()
        for relative_path, payload in closure.candidate_contents.items():
            files[f"{_SOURCE_ROOT}/{relative_path}"] = payload
        for name, payload in closure.operation_artifacts.items():
            files[f"{_AGENT_ARTIFACT_ROOT}/{name}"] = payload
        return dict(sorted(files.items()))

    @staticmethod
    def _parse_closure(payloads: Mapping[str, bytes]) -> ExpertCandidateClosure:
        manifest = ExpertCandidateManifest.from_json_bytes(payloads[_MANIFEST_PATH])
        tree = ExpertSourceTreeManifest.from_json_bytes(payloads[_SOURCE_TREE_PATH])
        modules = tuple(
            sorted(
                (
                    ExpertModuleContract.from_json_bytes(
                        payloads[f"{_MODULE_ROOT}/{module_ref.rsplit(':', 1)[1]}.json"]
                    )
                    for module_ref in manifest.module_contract_refs
                ),
                key=lambda module: module.module_id,
            )
        )
        candidate_contents = {
            file.relative_path: payloads[f"{_SOURCE_ROOT}/{file.relative_path}"]
            for file in tree.files
        }
        parent_files = ExpertCandidateStore._parse_contract_tuple(
            payloads[_PARENT_FILES_PATH],
            SourceFileDescriptor,
            "parent files",
        )
        ancestors = ExpertCandidateStore._parse_contract_tuple(
            payloads[_ANCESTORS_PATH],
            ExpertCandidateAncestorInput,
            "ancestor inputs",
        )
        closure = ExpertCandidateClosure(
            manifest=manifest,
            trigger_packet=ExpertTriggerEvidencePacket.from_json_bytes(
                payloads[_TRIGGER_PACKET_PATH]
            ),
            trigger_decision=ExpertEvolutionTriggerDecision.from_json_bytes(
                payloads[_TRIGGER_DECISION_PATH]
            ),
            patch=ExpertCandidatePatch.from_json_bytes(payloads[_PATCH_PATH]),
            candidate_tree=tree,
            parent_files=parent_files,
            repository_map=ExpertRepositoryMap.from_json_bytes(
                payloads[_REPOSITORY_MAP_PATH]
            ),
            module_contracts=modules,
            operation=ExpertCandidateOperationRecord.from_json_bytes(
                payloads[_OPERATION_PATH]
            ),
            workspace_delta=CodingAgentWorkspaceDelta.from_json_bytes(
                payloads[_WORKSPACE_DELTA_PATH]
            ),
            sanitation_report=ExpertCandidateSanitationReport.from_json_bytes(
                payloads[_SANITATION_PATH]
            ),
            candidate_contents=candidate_contents,
            operation_artifacts={
                name: payloads[f"{_AGENT_ARTIFACT_ROOT}/{name}"]
                for name in coding_agent_artifact_filenames(
                    CodingAgentWorkspaceAccess.EDIT_WORKSPACE
                )
            },
            ancestor_inputs=ancestors,
        )
        expected_payloads = ExpertCandidateStore._package_files(closure)
        if dict(payloads) != expected_payloads:
            raise ExpertCandidateStoreError(
                "candidate package differs from its canonical closure"
            )
        return closure

    @staticmethod
    def _contract_tuple_bytes(contracts: tuple[StrictContract, ...]) -> bytes:
        return canonical_json_bytes(tuple(contract.to_dict() for contract in contracts))

    @staticmethod
    def _parse_contract_tuple(
        payload: bytes,
        contract_type: type[StrictContract],
        name: str,
    ) -> tuple:
        parsed = parse_json_bytes(payload)
        if not isinstance(parsed, list):
            raise ExpertCandidateStoreError(f"candidate {name} must be an array")
        return tuple(contract_type.from_dict(item) for item in parsed)

    def _candidate_path(self, candidate_id: str) -> Path:
        require_content_id(candidate_id, "candidate_id")
        return self.object_root / candidate_id.rsplit(":", 1)[1]

    def _prepare_layout(self) -> None:
        if os.path.lexists(self.root):
            self._validate_private_directory(self.root, "candidate store root")
        else:
            os.mkdir(self.root, mode=0o700)
            self._fsync_directory(self.state_root)
        for path in (self.object_root, self.staging_root):
            if os.path.lexists(path):
                self._validate_private_directory(path, "candidate store directory")
            else:
                os.mkdir(path, mode=0o700)
        lock_path = self.root / "candidate.lock"
        if not os.path.lexists(lock_path):
            descriptor = os.open(
                lock_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
            )
            os.close(descriptor)
        self._validate_private_regular(lock_path, "candidate store lock")
        self._fsync_directory(self.root)

    def _recover_staging(self) -> None:
        entries = tuple(sorted(self.staging_root.iterdir()))
        for entry in entries:
            if _STAGING_PATTERN.fullmatch(entry.name) is None or entry.is_symlink():
                raise ExpertCandidateStoreError(
                    "candidate staging contains an invalid entry"
                )
            identity = restricted_directory_identity(
                self.staging_root,
                entry.name,
                ExpertCandidateStoreError,
            )
            remove_restricted_directory(
                self.staging_root,
                entry.name,
                identity,
                ExpertCandidateStoreError,
            )

    def _exclusive_lock(self):
        return _CandidateStoreLock(self.root / "candidate.lock", exclusive=True)

    def _shared_lock(self):
        return _CandidateStoreLock(self.root / "candidate.lock", exclusive=False)

    @staticmethod
    def _write_private_file(root: Path, relative_path: str, payload: bytes) -> None:
        path = PurePosixPath(relative_path)
        if (
            path.is_absolute()
            or path == PurePosixPath(".")
            or ".." in path.parts
            or path.as_posix() != relative_path
        ):
            raise ExpertCandidateStoreError("candidate package path is invalid")
        output = root.joinpath(*path.parts)
        current = root
        for part in path.parts[:-1]:
            current /= part
            if not os.path.lexists(current):
                os.mkdir(current, mode=0o700)
            ExpertCandidateStore._validate_private_directory(
                current,
                "candidate package directory",
            )
        descriptor = os.open(
            output,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _read_private_tree(root: Path) -> dict[str, bytes]:
        ExpertCandidateStore._validate_private_directory(
            root,
            "candidate package root",
        )
        with ExitStack() as descriptors:
            root_descriptor = os.open(
                root,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            )
            descriptors.callback(os.close, root_descriptor)
            opened = os.fstat(root_descriptor)
            opened_identity = (opened.st_dev, opened.st_ino)
            files = ExpertCandidateStore._read_private_directory(
                root_descriptor,
                PurePosixPath("."),
            )
            current = os.stat(root, follow_symlinks=False)
            if (current.st_dev, current.st_ino) != opened_identity:
                raise ExpertCandidateStoreError(
                    "candidate package root changed during read"
                )
        return files

    @staticmethod
    def _read_private_directory(
        directory_descriptor: int,
        relative_root: PurePosixPath,
    ) -> dict[str, bytes]:
        with os.scandir(directory_descriptor) as iterator:
            entries = tuple(
                sorted(
                    (
                        (entry.name, entry.stat(follow_symlinks=False))
                        for entry in iterator
                    ),
                    key=lambda item: item[0],
                )
            )
        files: dict[str, bytes] = {}
        for name, expected in entries:
            current = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            if (current.st_dev, current.st_ino) != (
                expected.st_dev,
                expected.st_ino,
            ):
                raise ExpertCandidateStoreError(
                    "candidate package entry changed during read"
                )
            relative_path = (
                PurePosixPath(name)
                if relative_root == PurePosixPath(".")
                else relative_root / name
            )
            if stat.S_ISDIR(expected.st_mode):
                if expected.st_mode & (0o077 | stat.S_ISUID | stat.S_ISGID):
                    raise ExpertCandidateStoreError(
                        "candidate package directory is not private"
                    )
                with ExitStack() as child_descriptors:
                    child_descriptor = os.open(
                        name,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                        dir_fd=directory_descriptor,
                    )
                    child_descriptors.callback(os.close, child_descriptor)
                    opened = os.fstat(child_descriptor)
                    if (opened.st_dev, opened.st_ino) != (
                        expected.st_dev,
                        expected.st_ino,
                    ):
                        raise ExpertCandidateStoreError(
                            "candidate package directory changed during read"
                        )
                    files.update(
                        ExpertCandidateStore._read_private_directory(
                            child_descriptor,
                            relative_path,
                        )
                    )
                continue
            if (
                not stat.S_ISREG(expected.st_mode)
                or expected.st_nlink != 1
                or expected.st_mode & (0o077 | stat.S_ISUID | stat.S_ISGID)
            ):
                raise ExpertCandidateStoreError(
                    "candidate package entry is not a private independent file"
                )
            descriptor = os.open(
                name,
                os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=directory_descriptor,
            )
            with os.fdopen(descriptor, "rb") as handle:
                opened = os.fstat(handle.fileno())
                if (
                    (opened.st_dev, opened.st_ino) != (expected.st_dev, expected.st_ino)
                    or not stat.S_ISREG(opened.st_mode)
                    or opened.st_nlink != 1
                    or opened.st_mode & (0o077 | stat.S_ISUID | stat.S_ISGID)
                ):
                    raise ExpertCandidateStoreError(
                        "candidate package file changed during read"
                    )
                files[relative_path.as_posix()] = handle.read()
        return files

    @staticmethod
    def _validate_state_root(path: Path) -> None:
        if (
            not path.is_absolute()
            or path != Path(os.path.abspath(path))
            or path in {Path("/"), Path.home()}
            or path.is_symlink()
            or not path.is_dir()
            or path.resolve() != path
        ):
            raise ExpertCandidateStoreError(
                "candidate state root must be an authorized real directory"
            )
        if path.stat().st_mode & (0o077 | stat.S_ISUID | stat.S_ISGID):
            raise ExpertCandidateStoreError("candidate state root must be private")

    @staticmethod
    def _validate_private_directory(path: Path, name: str) -> None:
        if path.is_symlink() or not path.is_dir():
            raise ExpertCandidateStoreError(f"{name} must be a real directory")
        metadata = path.stat(follow_symlinks=False)
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_mode & (
            0o077 | stat.S_ISUID | stat.S_ISGID
        ):
            raise ExpertCandidateStoreError(f"{name} must be a private directory")

    @staticmethod
    def _validate_private_regular(path: Path, name: str) -> None:
        if path.is_symlink() or not path.is_file():
            raise ExpertCandidateStoreError(f"{name} must be a regular file")
        metadata = path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_mode & (0o077 | stat.S_ISUID | stat.S_ISGID)
        ):
            raise ExpertCandidateStoreError(
                f"{name} must be a private independent file"
            )

    @staticmethod
    def _fsync_tree(root: Path) -> None:
        directories = tuple(
            sorted(
                (path for path in root.rglob("*") if path.is_dir()),
                key=lambda path: len(path.parts),
                reverse=True,
            )
        )
        for directory in directories:
            ExpertCandidateStore._fsync_directory(directory)
        ExpertCandidateStore._fsync_directory(root)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        os.fsync(descriptor)
        os.close(descriptor)

    @staticmethod
    def _rename_directory_no_replace(source: Path, destination: Path) -> None:
        libc = ctypes.CDLL(None, use_errno=True)
        if not hasattr(libc, "renameat2"):
            raise ExpertCandidateStoreError(
                "atomic no-replace candidate publication is unavailable"
            )
        rename_at2 = libc.renameat2
        rename_at2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename_at2.restype = ctypes.c_int
        result = rename_at2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            _RENAME_NOREPLACE,
        )
        if result != 0:
            error_number = ctypes.get_errno()
            raise OSError(
                error_number,
                f"candidate publication failed: {errno.errorcode.get(error_number)}",
            )


class _CandidateStoreLock:
    def __init__(self, path: Path, *, exclusive: bool, create: bool = False):
        self.path = path
        self.exclusive = exclusive
        self.create = create
        self.handle = None

    def __enter__(self):
        flags = os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC
        if self.create:
            flags |= os.O_CREAT
        descriptor = os.open(
            self.path,
            flags,
            0o600,
        )
        self.handle = os.fdopen(descriptor, "r+b")
        metadata = os.fstat(self.handle.fileno())
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_mode & (0o077 | stat.S_ISUID | stat.S_ISGID)
        ):
            self.handle.close()
            raise ExpertCandidateStoreError(
                "candidate lock must be a private independent file"
            )
        fcntl.flock(
            self.handle.fileno(),
            fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH,
        )
        return self

    def __exit__(self, exception_type, exception, traceback):
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()
        self.handle = None
        return False
