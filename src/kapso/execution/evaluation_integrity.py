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


def build_data_manifest(
    root: str | Path, paths: "list[str]"
) -> Dict[str, str]:
    """Hash the protected evaluation-input files under workspace paths.

    Evaluation identity has two halves: the logic (the evaluation tree,
    fingerprinted into evaluator_id) and the inputs it reads. This is the
    inputs half — the files a candidate must never rewrite, because a
    score measured on rewritten inputs belongs to a different evaluation
    set while sitting in the same comparability class.
    """
    base = Path(root).expanduser()
    manifest: Dict[str, str] = {}
    for raw_path in paths:
        target = base / raw_path
        if not target.exists():
            raise EvaluationIntegrityError(
                f"Protected data path does not exist: {target}"
            )
        files = (
            sorted(target.rglob("*")) if target.is_dir() else [target]
        )
        for path in files:
            if path.is_symlink():
                raise EvaluationIntegrityError(
                    f"Protected data cannot contain symlinks: {path}"
                )
            if path.is_file():
                relative = path.relative_to(base).as_posix()
                manifest[relative] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    return manifest


def verify_data_manifest(
    root: str | Path, expected: Mapping[str, str]
) -> str:
    """Empty string when every protected input matches; else the defect."""
    base = Path(root)
    problems = []
    for relative, digest in sorted(expected.items()):
        path = base / relative
        if not path.is_file():
            problems.append(f"missing:{relative}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            problems.append(f"modified:{relative}")
    if problems:
        return (
            "Protected evaluation data changed by the candidate: "
            + ", ".join(problems)
        )
    return ""
