"""Deterministic repository memory rebuilt from one pinned launch workspace."""

from __future__ import annotations

import re
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import PurePosixPath

from kapso.cross_run.canonical import canonical_json_bytes, tree_or_blob_digest
from kapso.cross_run.git_command import BoundedGitCommand
from kapso.cross_run.github.command import CommandOutputKind
from kapso.cross_run.launch.workspace import ActiveLaunchWorkspace
from kapso.cross_run.launch.workspace_frontier import inspect_run_workspace_frontier
from kapso.cross_run.settings import CrossRunSettings


class RepositoryMemoryError(RuntimeError):
    """The pinned repository cannot produce one exact non-mutating memory view."""


@dataclass(frozen=True)
class RepositoryMemory:
    """Complete canonical repository map safe to include in coding-agent context."""

    payload: bytes
    digest: str
    source_commit_sha: str
    source_tree_sha: str

    def __post_init__(self) -> None:
        if (
            type(self.payload) is not bytes
            or not self.payload
            or self.digest != tree_or_blob_digest(self.payload)
            or re.fullmatch(r"[0-9a-f]{40}", self.source_commit_sha) is None
            or re.fullmatch(r"[0-9a-f]{40}", self.source_tree_sha) is None
        ):
            raise RepositoryMemoryError("repository memory identity is invalid")


def build_repository_memory(
    *,
    active_workspace: ActiveLaunchWorkspace,
    settings: CrossRunSettings,
) -> RepositoryMemory:
    """Read the exact baseline commit and emit a deterministic book of contents."""

    if (
        type(active_workspace) is not ActiveLaunchWorkspace
        or type(settings) is not CrossRunSettings
    ):
        raise RepositoryMemoryError(
            "repository memory requires exact launch and configuration authority"
        )
    active_workspace.require_control_authority()
    installation = active_workspace.bootstrap_pin.installation_receipt
    expected_commit = installation.workspace_baseline_commit_sha
    before = _inspect_workspace(active_workspace, settings, expected_commit)
    command = BoundedGitCommand(
        timeout_seconds=settings.capture.git_command_timeout_seconds,
        maximum_output_bytes=settings.capture.git_command_output_bytes,
    )
    listing = command.run(
        active_workspace.workspace,
        ("ls-tree", "-r", "-l", "-z", expected_commit),
        output_kind=CommandOutputKind.BINARY,
    )
    if listing.returncode != 0 or listing.stderr:
        raise RepositoryMemoryError("repository memory Git listing failed")
    files = tuple(
        _parse_tree_entry(entry) for entry in listing.stdout.split(b"\0") if entry
    )
    if (
        not files
        or len(files) > settings.launch.run_workspace_entry_limit
        or sum(file[2] for file in files) > settings.launch.run_workspace_size_bytes
    ):
        raise RepositoryMemoryError(
            "repository memory source closure is empty or exceeds configured bounds"
        )
    paths = tuple(file[0] for file in files)
    payload = canonical_json_bytes(
        {
            "schema_version": "kapso.repository_memory.v1",
            "summary": (
                f"Pinned repository with {len(files)} regular source files across "
                f"{len({PurePosixPath(path).parts[0] for path in paths})} top-level areas."
            ),
            "source_commit_sha": expected_commit,
            "source_tree_sha": installation.workspace_baseline_tree_sha,
            "source_composition_hash": installation.expected_source_composition_hash,
            "table_of_contents": (_repository_table_of_contents(paths)),
            "files": tuple(
                {"path": path, "mode": mode, "size_bytes": size}
                for path, mode, size in files
            ),
        }
    )
    if len(payload) > settings.launch.run_workspace_git_metadata_size_bytes:
        raise RepositoryMemoryError(
            "repository memory exceeds its configured metadata bound"
        )
    if _inspect_workspace(active_workspace, settings, expected_commit) != before:
        raise RepositoryMemoryError(
            "pinned workspace changed while rebuilding repository memory"
        )
    return RepositoryMemory(
        payload=payload,
        digest=tree_or_blob_digest(payload),
        source_commit_sha=expected_commit,
        source_tree_sha=installation.workspace_baseline_tree_sha,
    )


def _inspect_workspace(
    active_workspace: ActiveLaunchWorkspace,
    settings: CrossRunSettings,
    expected_commit: str,
):
    with ExitStack() as descriptors:
        workspace_descriptor, _identity = active_workspace._open_execution_workspace(
            descriptors
        )
        return inspect_run_workspace_frontier(
            workspace_descriptor,
            settings=settings.launch,
            expected_commit_sha=expected_commit,
        )


def _parse_tree_entry(entry: bytes) -> tuple[str, str, int]:
    metadata, separator, encoded_path = entry.partition(b"\t")
    fields = metadata.decode("ascii").split()
    path = encoded_path.decode("utf-8")
    normalized = PurePosixPath(path)
    if (
        separator != b"\t"
        or len(fields) != 4
        or fields[0] not in {"100644", "100755"}
        or fields[1] != "blob"
        or re.fullmatch(r"[0-9a-f]{40}", fields[2]) is None
        or not fields[3].isdigit()
        or normalized.is_absolute()
        or normalized == PurePosixPath(".")
        or ".." in normalized.parts
        or normalized.as_posix() != path
    ):
        raise RepositoryMemoryError(
            "repository memory Git tree entry is unsupported or unsafe"
        )
    return path, fields[0], int(fields[3])


def _repository_table_of_contents(paths: tuple[str, ...]) -> tuple[dict, ...]:
    sections = (
        (
            "repository.entrypoints",
            "Entrypoints",
            tuple(
                path
                for path in paths
                if PurePosixPath(path).name
                in {
                    "Dockerfile",
                    "Makefile",
                    "main.py",
                    "pyproject.toml",
                    "setup.py",
                }
            ),
        ),
        (
            "repository.tests",
            "Tests",
            tuple(
                path
                for path in paths
                if "test" in PurePosixPath(path).name.lower()
                or "tests" in PurePosixPath(path).parts
            ),
        ),
        (
            "repository.documentation",
            "Documentation",
            tuple(
                path
                for path in paths
                if PurePosixPath(path).suffix.lower() == ".md"
                or "docs" in PurePosixPath(path).parts
            ),
        ),
        (
            "repository.top_level",
            "Top-level areas",
            tuple(sorted({PurePosixPath(path).parts[0] for path in paths})),
        ),
    )
    return tuple(
        {
            "section_id": section_id,
            "title": title,
            "paths": section_paths,
        }
        for section_id, title, section_paths in sections
    )


__all__ = [
    "build_repository_memory",
    "RepositoryMemory",
    "RepositoryMemoryError",
]
