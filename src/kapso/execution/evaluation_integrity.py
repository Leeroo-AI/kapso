"""Integrity checks for caller-provided evaluation suites."""

from __future__ import annotations

import hashlib
import json
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional


PROVIDED = "provided"
AGENT_GENERATED = "agent_generated"
VALID_PROVENANCE = {PROVIDED, AGENT_GENERATED}

# New runtime outputs are allowed, but new evaluator source/config files are
# suspicious when the caller supplied the suite that defines the score.
PROTECTED_SUFFIXES = {
    ".c",
    ".cc",
    ".cfg",
    ".cpp",
    ".go",
    ".java",
    ".js",
    ".jsx",
    ".ipynb",
    ".ini",
    ".pl",
    ".ps1",
    ".py",
    ".r",
    ".rb",
    ".rs",
    ".sh",
    ".bash",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".yaml",
    ".yml",
    ".zsh",
}


class EvaluationIntegrityError(RuntimeError):
    """Raised when a provided evaluation source cannot be fingerprinted."""


@dataclass(frozen=True)
class EvaluationIntegrityReport:
    """Result of comparing a candidate suite with the provided baseline."""

    valid: bool
    fingerprint: Optional[str]
    error: str = ""


def build_evaluation_manifest(directory: str | Path) -> Dict[str, str]:
    """Hash every regular file in a provided evaluation directory."""
    root = Path(directory).expanduser()
    if not root.exists():
        raise FileNotFoundError(
            f"Provided evaluation directory does not exist: {root}"
        )
    if root.is_symlink():
        raise EvaluationIntegrityError(
            f"Provided evaluation directory cannot be a symlink: {root}"
        )
    if not root.is_dir():
        raise NotADirectoryError(
            f"Provided evaluation path is not a directory: {root}"
        )

    manifest: Dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise EvaluationIntegrityError(
                "Provided evaluation trees cannot contain symlinks: "
                f"{relative}"
            )
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest[relative] = digest

    if not manifest:
        raise EvaluationIntegrityError(
            f"Provided evaluation directory contains no files: {root}"
        )
    return manifest


def manifest_fingerprint(manifest: Mapping[str, str]) -> str:
    """Return a deterministic fingerprint for a file manifest."""
    encoded = json.dumps(
        dict(manifest),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_evaluation_tree(
    directory: str | Path,
    expected_manifest: Mapping[str, str],
) -> EvaluationIntegrityReport:
    """Verify protected files and reject newly added evaluator source."""
    try:
        current = build_evaluation_manifest(directory)
    except Exception as exc:
        return EvaluationIntegrityReport(
            valid=False,
            fingerprint=None,
            error=f"{type(exc).__name__}: {exc}",
        )

    missing = sorted(set(expected_manifest) - set(current))
    changed = sorted(
        path
        for path, digest in expected_manifest.items()
        if current.get(path) not in {None, digest}
    )
    unexpected_source = sorted(
        path
        for path in set(current) - set(expected_manifest)
        if (
            Path(path).suffix.lower() in PROTECTED_SUFFIXES
            or (
                Path(directory, path).stat().st_mode
                & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            )
        )
    )

    problems = []
    if missing:
        problems.append("missing: " + ", ".join(missing))
    if changed:
        problems.append("changed: " + ", ".join(changed))
    if unexpected_source:
        problems.append(
            "unexpected evaluator source: "
            + ", ".join(unexpected_source)
        )
    if problems:
        return EvaluationIntegrityReport(
            valid=False,
            fingerprint=manifest_fingerprint(current),
            error="Provided evaluation integrity check failed ("
            + "; ".join(problems)
            + ")",
        )
    return EvaluationIntegrityReport(
        valid=True,
        fingerprint=manifest_fingerprint(current),
    )
