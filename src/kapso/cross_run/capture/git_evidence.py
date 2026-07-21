"""Pure verification helpers for self-contained Git commit and tree evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

from kapso.cross_run.git_refs import git_object_sha

_GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class ParsedCommit:
    tree_sha: str
    parent_shas: tuple[str, ...]


def parse_commit_object(payload: bytes) -> ParsedCommit:
    """Read the tree and parent headers from an unframed Git commit payload."""
    header = payload.split(b"\n\n", 1)[0]
    lines = header.splitlines()
    tree_values = tuple(line[5:] for line in lines if line.startswith(b"tree "))
    parent_values = tuple(line[7:] for line in lines if line.startswith(b"parent "))
    if len(tree_values) != 1:
        raise ValueError("Git commit must name exactly one root tree")
    values = (*tree_values, *parent_values)
    if any(
        len(value) != 40
        or _GIT_SHA_PATTERN.fullmatch(value.decode("ascii", errors="strict")) is None
        for value in values
    ):
        raise ValueError("Git commit contains an invalid object id")
    return ParsedCommit(
        tree_sha=tree_values[0].decode("ascii"),
        parent_shas=tuple(value.decode("ascii") for value in parent_values),
    )


def reconstruct_root_tree_sha(
    entries: Mapping[str, tuple[str, str]],
) -> str:
    """Reconstruct a Git root-tree id from its complete recursive leaf listing."""
    if not entries:
        return git_object_sha("tree", b"")
    trie: dict[bytes, object] = {}
    for path, (mode, object_sha) in entries.items():
        encoded_parts = tuple(part.encode("utf-8") for part in path.split("/"))
        if any(not part for part in encoded_parts):
            raise ValueError("Git tree entry path is invalid")
        branch = trie
        for part in encoded_parts[:-1]:
            existing = branch.get(part)
            if existing is None:
                child: dict[bytes, object] = {}
                branch[part] = child
                branch = child
            elif isinstance(existing, dict):
                branch = existing
            else:
                raise ValueError("Git tree contains a file/directory path collision")
        leaf = encoded_parts[-1]
        if leaf in branch:
            raise ValueError("Git tree entry path is duplicated")
        if _GIT_SHA_PATTERN.fullmatch(object_sha) is None:
            raise ValueError("Git tree leaf object id is invalid")
        branch[leaf] = (mode, object_sha)

    def tree_sha(branch: dict[bytes, object]) -> str:
        materialized: list[tuple[bytes, bytes]] = []
        for name, value in branch.items():
            if isinstance(value, dict):
                mode = "40000"
                object_sha = tree_sha(value)
                sort_key = name + b"/"
            else:
                mode, object_sha = value
                sort_key = name
            entry = (
                mode.encode("ascii") + b" " + name + b"\0" + bytes.fromhex(object_sha)
            )
            materialized.append((sort_key, entry))
        payload = b"".join(entry for _, entry in sorted(materialized))
        return git_object_sha("tree", payload)

    return tree_sha(trie)


def has_ancestry_path(
    commit_graph: Mapping[str, tuple[str, ...]],
    descendant_sha: str,
    ancestor_sha: str,
) -> bool:
    """Return whether the supplied commit-object closure proves ancestry."""
    pending = [descendant_sha]
    visited: set[str] = set()
    while pending:
        current = pending.pop()
        if current == ancestor_sha:
            return True
        if current in visited:
            continue
        visited.add(current)
        pending.extend(commit_graph.get(current, ()))
    return False
