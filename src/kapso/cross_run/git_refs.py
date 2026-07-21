"""Git reference validation shared by configuration and publication paths."""

from __future__ import annotations

import hashlib
import re
from pathlib import PurePosixPath
from typing import Mapping

_SAFE_REF_PATTERN = re.compile(r"[A-Za-z0-9._/-]+")


def git_object_sha(object_kind: str, payload: bytes) -> str:
    """Return the Git SHA-1 identity for canonical object bytes."""
    if object_kind not in {"blob", "commit", "tree"} or not isinstance(payload, bytes):
        raise ValueError("Git object identity input is invalid")
    header = f"{object_kind} {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload, usedforsecurity=False).hexdigest()


def git_tree_shas(files: Mapping[str, tuple[str, str]]) -> dict[str, str]:
    """Recompute every Git tree SHA from an exact flat regular-file closure."""
    if not files:
        raise ValueError("Git tree file closure must not be empty")
    file_entries: dict[tuple[str, ...], list[tuple[bytes, str, str]]] = {}
    directories: set[tuple[str, ...]] = {()}
    seen_paths: set[tuple[str, ...]] = set()
    for relative_path, (blob_sha, mode) in files.items():
        path = PurePosixPath(relative_path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != relative_path
            or relative_path == "."
            or not re.fullmatch(r"[0-9a-f]{40}", blob_sha)
            or mode not in {"100644", "100755"}
        ):
            raise ValueError("Git tree file descriptor is invalid")
        parent = path.parts[:-1]
        if path.parts in seen_paths:
            raise ValueError("Git tree file closure contains a duplicate path")
        seen_paths.add(path.parts)
        file_entries.setdefault(parent, []).append(
            (path.parts[-1].encode("utf-8"), mode, blob_sha)
        )
        for depth in range(len(path.parts)):
            directories.add(path.parts[:depth])
    child_directories: dict[tuple[str, ...], list[tuple[str, ...]]] = {}
    for directory in directories:
        if directory:
            child_directories.setdefault(directory[:-1], []).append(directory)
    tree_shas: dict[tuple[str, ...], str] = {}
    for directory in sorted(directories, key=lambda value: (-len(value), value)):
        entries = list(file_entries.get(directory, ()))
        entries.extend(
            (child[-1].encode("utf-8") + b"/", "40000", tree_shas[child])
            for child in child_directories.get(directory, ())
        )
        payload = b"".join(
            mode.encode("ascii")
            + b" "
            + sort_name.removesuffix(b"/")
            + b"\0"
            + bytes.fromhex(object_sha)
            for sort_name, mode, object_sha in sorted(entries)
        )
        tree_shas[directory] = git_object_sha("tree", payload)
    return {
        PurePosixPath(*directory).as_posix() if directory else "": object_sha
        for directory, object_sha in tree_shas.items()
    }


def require_git_ref_name(
    value: str,
    name: str,
    *,
    qualified: bool,
    error_type: type[Exception],
) -> str:
    """Require the safe subset of Git's ref-name grammar used by Kapso."""
    parts = value.split("/") if isinstance(value, str) else []
    if (
        not isinstance(value, str)
        or not _SAFE_REF_PATTERN.fullmatch(value)
        or (qualified and (len(parts) < 3 or parts[0] != "refs"))
        or (not qualified and (not parts or len(parts) != 1))
        or any(
            not part
            or part.startswith(".")
            or part.endswith(".")
            or part.endswith(".lock")
            for part in parts
        )
        or ".." in value
        or "@{" in value
    ):
        raise error_type(f"{name} is not a valid Git ref name")
    return value
