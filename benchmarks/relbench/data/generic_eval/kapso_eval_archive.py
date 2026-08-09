"""Sandbox half of the evaluation-archive contract — vendored into suites.

This file is the FRAMEWORK MASTER of the archive mechanics that must run
inside candidate sandboxes, where kapso is not importable: benchmark
evaluation suites ship a byte-identical copy next to their grader (pinned by
test_evaluation_archive.test_vendored_copies_are_byte_identical) and import
it as a sibling module. It is therefore deliberately stdlib-only and
self-contained.

It provides the recording side of evaluation governance:

- ``fingerprint_tree``   — the evaluator identity stamp. Mirrors
  ``evaluation_integrity.build_evaluation_manifest`` +
  ``manifest_fingerprint`` exactly, except that runtime junk (__pycache__,
  *.pyc) is excluded so a tree that has merely been EXECUTED still hashes to
  its registered fingerprint. The mirror is pinned by an agreement test; do
  not change one side without the other.
- ``allocate_run_dir``   — race-free run_%04d allocation under a lock file.
- ``write_selection_label`` / ``void_run`` — the selection-eligibility labels
  final selection consumes.
- ``snapshot_evaluator_tree`` — one junk-free copy of the evaluation tree per
  evaluator version, which is what makes the version's scoring logic
  reachable after the campaign without the strategy workspace.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import shutil
from pathlib import Path

MANIFEST_MARKER = "KAPSO_EVAL_MANIFEST"
RUNTIME_JUNK_DIRS = {"__pycache__"}
RUNTIME_JUNK_SUFFIXES = {".pyc"}


def _is_runtime_junk(relative: Path) -> bool:
    return (
        bool(RUNTIME_JUNK_DIRS.intersection(relative.parts))
        or relative.suffix.lower() in RUNTIME_JUNK_SUFFIXES
    )


def tree_manifest(directory: Path) -> dict:
    """Hash every regular non-junk file, mirroring the framework manifest."""
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Evaluation tree does not exist or is not a directory: {root}"
        )
    manifest = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if _is_runtime_junk(relative):
            continue
        if path.is_symlink():
            raise ValueError(
                f"Evaluation trees cannot contain symlinks: {relative.as_posix()}"
            )
        if not path.is_file():
            continue
        manifest[relative.as_posix()] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    if not manifest:
        raise ValueError(f"Evaluation tree contains no files: {root}")
    return manifest


def fingerprint_tree(directory: Path) -> str:
    """The evaluator identity of a tree — equals the registered fingerprint."""
    encoded = json.dumps(
        tree_manifest(directory), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def allocate_run_dir(runs_root: Path) -> Path:
    """Create and return the next runs/run_%04d directory, race-free.

    Parallel sessions archive concurrently; the previous glob-then-mkdir
    pattern could hand two sessions the same index. The lock file serializes
    allocation; the directory itself is created inside the critical section.
    """
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    lock_path = runs_root / ".allocate.lock"
    with open(lock_path, "w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing = sorted(runs_root.glob("run_*"))
        next_index = (
            int(existing[-1].name.split("_")[1]) + 1 if existing else 1
        )
        run_dir = runs_root / f"run_{next_index:04d}"
        (run_dir / "private").mkdir(parents=True)
        fcntl.flock(lock, fcntl.LOCK_UN)
    return run_dir


def write_selection_label(
    run_dir: Path, *, session: str, evaluator_id: str
) -> None:
    """Every archive starts pending, stamped with the evaluator that made it.

    The stamp is what final selection pools on: runs measured under a
    superseded evaluator version never compete with head-measured runs,
    mirroring the in-loop rule that no score crosses an evaluator_id
    boundary.
    """
    (Path(run_dir) / "private" / "selection.json").write_text(
        json.dumps(
            {
                "status": "pending",
                "session": session,
                "by": "grader",
                "evaluator_id": evaluator_id,
            },
            indent=2,
        )
    )


def snapshot_evaluator_tree(
    archive_root: Path, evaluation_dir: Path, evaluator_id: str
) -> Path:
    """Persist one junk-free copy of the tree per evaluator version.

    Idempotent: the first archive under a fingerprint writes the snapshot,
    later ones find it present. The copy excludes runtime junk so
    re-fingerprinting the snapshot reproduces ``evaluator_id`` — final
    selection verifies exactly that before trusting the snapshot's scoring
    logic.
    """
    destination = Path(archive_root) / "evaluators" / evaluator_id
    if destination.exists():
        return destination
    staging = destination.with_name(destination.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    source = Path(evaluation_dir)
    for relative in sorted(tree_manifest(source)):
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
    staging.replace(destination)
    return destination


def void_run(
    runs_root: Path, run_name: str, *, session: str, reason: str
) -> dict:
    """Candidate-initiated disqualification of one of ITS OWN archived runs.

    Cross-session voids are refused: a session may only retract its own
    work. Returns the updated label record for the caller to report.
    """
    if not reason.strip():
        raise ValueError("void requires a non-empty reason")
    label_path = Path(runs_root) / run_name / "private" / "selection.json"
    if not label_path.exists():
        raise FileNotFoundError(
            f"no archived run {run_name!r} with a selection label"
        )
    record = json.loads(label_path.read_text())
    if record["session"] != session:
        raise PermissionError(
            f"void refused: {run_name} belongs to session "
            f"{record['session']!r}, not {session!r}"
        )
    record.update(
        {"status": "self-voided", "by": "candidate", "reason": reason.strip()}
    )
    label_path.write_text(json.dumps(record, indent=2))
    return record
