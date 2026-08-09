"""Host half of the evaluation-archive contract: labels, head, final selection.

Framework owns HOW measurements are recorded, labeled, versioned, and
selected; a benchmark owns WHAT a measurement is. This module is the
selection side of that split (EVALUATION_GOVERNANCE.md): it reads the archive
a suite's grader wrote through ``evaluation_archive_sandbox`` and produces
the campaign's shipped pick under the MOST RECENT evaluator version, with the
same doctrine the in-loop search applies to ``node.score``:

- no score crosses an evaluator_id boundary — runs archived under a
  superseded evaluator are excluded, never re-ranked;
- missing evidence never wins — an unmeasurable run cannot outrank a
  head-measured one;
- stored scores are never trusted — ranking values are recomputed from the
  stored predictions by the head evaluator's own ``--rescore`` mode, and the
  archive-time score is only a tamper tripwire against the recomputation.

Everything verification-shaped fails loud (no fallbacks): a missing stamp,
missing snapshot, fingerprint mismatch, failing rescore, or tripwire
disagreement raises. The one non-exceptional outcome — an archive with no
finals under the head — returns an empty selection for the caller to report.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from kapso.execution.evaluation_archive_sandbox import fingerprint_tree
from kapso.execution.evaluation_integrity import (
    build_evaluation_manifest,
    manifest_fingerprint,
)
from kapso.execution.evaluation_maintainer.maintainer import (
    ENTRYPOINT_NAME,
    parse_manifest_line,
)

RESCORE_TOLERANCE = {"rel_tol": 1e-9, "abs_tol": 1e-12}


@dataclass(frozen=True)
class FinalSelection:
    """The shipped pick and its full provenance."""

    head_evaluator_id: str
    winner_run: Optional[str]
    winner_score: Optional[float]
    winner_metrics: Dict[str, float] = field(default_factory=dict)
    scored: Dict[str, float] = field(default_factory=dict)
    excluded: Dict[str, str] = field(default_factory=dict)


def _selection_record(run_dir: Path) -> Dict:
    label_path = run_dir / "private" / "selection.json"
    if not label_path.exists():
        raise FileNotFoundError(
            f"{run_dir.name} has no selection label — archive was written "
            "by a pre-label grader generation"
        )
    return json.loads(label_path.read_text())


def infer_session_finals(runs_root: Path) -> None:
    """Resolve each session's registered final from the archive itself.

    ``finalize_session_run`` stamps labels when the search hands over a
    manifest of record, but the manifest is printed by the evaluation wrapper
    the maintainer owns and does not always carry run/session identity.
    selection.json always does (written at archive time), and a session's
    LAST archived run is its registered result — after an evaluator
    transition that is the bridged re-measurement, so finals migrate to the
    new evaluator without any separate synchronization. Never overrides a
    decided label: self-voided, invalid, and already-final runs are left
    untouched.
    """
    by_session: Dict[str, List[Path]] = {}
    for run_dir in sorted(Path(runs_root).glob("run_*")):
        record = _selection_record(run_dir)
        if record["status"] != "pending":
            continue
        by_session.setdefault(record["session"], []).append(run_dir)
    for session, runs in by_session.items():
        final = max(runs, key=lambda r: r.name)
        for run_dir in runs:
            label_path = run_dir / "private" / "selection.json"
            record = json.loads(label_path.read_text())
            record["status"] = "final" if run_dir is final else "superseded"
            record["by"] = "session-final inference"
            label_path.write_text(json.dumps(record, indent=2))
        print(
            f"[evaluation-archive] session {session}: final={final.name} "
            f"(+{len(runs) - 1} superseded)"
        )


def finalize_session_run(runs_root: Path, manifest: Mapping, valid: bool) -> None:
    """Label the archive when a session's score of record resolves.

    The of-record run (named in the manifest the grader printed) becomes the
    session's registered "final" — or "invalid" on a judge veto or integrity
    flag; every still-pending sibling from the same session is "superseded".
    A run the candidate already self-voided keeps that stamp: a session's
    retraction of its own work outranks promotion.
    """
    if manifest["fidelity"] != "full":
        return
    run_name, session = manifest["run"], manifest["session"]
    if not run_name:
        raise ValueError(
            "full-fidelity manifest of record carries no run name — "
            "grader/archive generation mismatch"
        )
    runs_root = Path(runs_root)
    label_path = runs_root / run_name / "private" / "selection.json"
    record = json.loads(label_path.read_text())
    if record["status"] == "self-voided":
        print(
            f"[evaluation-archive] run selection: {run_name} stays "
            "self-voided (candidate retraction outranks promotion)"
        )
    else:
        record.update(
            {"status": "final" if valid else "invalid", "by": "strategy"}
        )
        label_path.write_text(json.dumps(record, indent=2))
    superseded = 0
    for other in sorted(runs_root.glob("run_*")):
        other_label = other / "private" / "selection.json"
        if other.name == run_name or not other_label.exists():
            continue
        other_record = json.loads(other_label.read_text())
        if (
            other_record["session"] == session
            and other_record["status"] == "pending"
        ):
            other_record.update({"status": "superseded", "by": "strategy"})
            other_label.write_text(json.dumps(other_record, indent=2))
            superseded += 1
    print(
        f"[evaluation-archive] run selection: {run_name} -> "
        f"{record['status']}; superseded {superseded} sibling run(s) "
        f"of session {session}"
    )


def resolve_head(runs_root: Path) -> str:
    """The most recent evaluator version = the stamp on the newest archive.

    Every validly scored archive was produced by the registered head of its
    moment (per-candidate integrity enforcement guarantees the tree matched
    the registered manifest), and the registry only moves forward — so the
    newest archive carries the current head. An accepted mid-run change
    produces head-stamped archives immediately via the transition bridge.
    """
    runs = sorted(Path(runs_root).glob("run_*"))
    if not runs:
        raise FileNotFoundError(f"archive has no runs: {runs_root}")
    newest = runs[-1]
    record = _selection_record(newest)
    head = record.get("evaluator_id", "")
    if not head:
        raise ValueError(
            f"{newest.name} carries no evaluator stamp — archive predates "
            "the governance contract; re-run the campaign to produce "
            "stamped archives"
        )
    return head


def verify_snapshot(
    archive_root: Path,
    evaluator_id: str,
    provided_files: Optional[Mapping[str, str]] = None,
) -> Path:
    """The snapshot must hash to its name before its scoring logic is run.

    ``provided_files`` byte-anchors the immutable provided core (relpath ->
    sha256 from the caller's own shipped copy): whatever the maintainer
    evolved around it, the provided scoring logic inside the snapshot must be
    exactly what the benchmark shipped.
    """
    snapshot = Path(archive_root) / "evaluators" / evaluator_id
    if not snapshot.is_dir():
        raise FileNotFoundError(
            f"no evaluator snapshot for head {evaluator_id[:12]} under "
            f"{snapshot.parent}"
        )
    manifest = build_evaluation_manifest(snapshot)
    actual = manifest_fingerprint(manifest)
    if actual != evaluator_id:
        raise ValueError(
            f"evaluator snapshot {evaluator_id[:12]} hashes to "
            f"{actual[:12]} — snapshot was modified after archival"
        )
    for relative, digest in sorted((provided_files or {}).items()):
        if manifest.get(relative) != digest:
            raise ValueError(
                f"provided evaluator file {relative!r} inside snapshot "
                f"{evaluator_id[:12]} does not match the shipped copy"
            )
    return snapshot


def rescore_run(
    snapshot: Path, run_dir: Path, env: Optional[Dict[str, str]] = None
) -> Dict:
    """Recompute one run's score of record with the head evaluator.

    Executes no candidate code: the entrypoint's ``--rescore`` mode scores
    the run's STORED artifacts. The archive-time manifest is then used as a
    tamper tripwire — a stored score that disagrees with its own
    recomputation means the archive was edited or the scoring drifted, and
    either one is fatal rather than silently resolvable.
    """
    entrypoint = Path(snapshot) / ENTRYPOINT_NAME
    if not entrypoint.exists():
        raise FileNotFoundError(
            f"evaluator snapshot has no {ENTRYPOINT_NAME} — the registered "
            "entrypoint contract requires it"
        )
    completed = subprocess.run(
        [sys.executable, str(entrypoint), "--rescore", str(run_dir)],
        cwd=str(snapshot),
        env=dict(env) if env is not None else os.environ.copy(),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"rescore of {Path(run_dir).name} failed "
            f"(exit {completed.returncode}): "
            f"{completed.stdout[-2000:]}{completed.stderr[-2000:]}"
        )
    payload = parse_manifest_line(completed.stdout)

    stored_path = Path(run_dir) / "manifest.txt"
    if not stored_path.exists():
        raise FileNotFoundError(
            f"{Path(run_dir).name} has no archived manifest line to "
            "cross-check the rescore against"
        )
    stored = parse_manifest_line(stored_path.read_text())
    if not math.isclose(
        float(payload["score"]), float(stored["score"]), **RESCORE_TOLERANCE
    ):
        raise ValueError(
            f"{Path(run_dir).name}: archived score {stored['score']} does "
            f"not match its recomputation {payload['score']} — the archive "
            "was edited or the evaluator's scoring drifted"
        )
    return payload


def select_final(
    archive_root: Path,
    *,
    higher_is_better: bool,
    provided_files: Optional[Mapping[str, str]] = None,
    expected_head_id: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> FinalSelection:
    """Max rescored validation among head-stamped finals — the shipped pick."""
    archive_root = Path(archive_root)
    runs_root = archive_root / "runs"
    infer_session_finals(runs_root)

    head = resolve_head(runs_root)
    if expected_head_id is not None and head != expected_head_id:
        raise ValueError(
            f"archive head {head[:12]} does not match the expected "
            f"registered head {expected_head_id[:12]}"
        )
    snapshot = verify_snapshot(archive_root, head, provided_files)

    scored: Dict[str, float] = {}
    excluded: Dict[str, str] = {}
    best: Optional[str] = None
    best_payload: Dict = {}
    for run_dir in sorted(runs_root.glob("run_*")):
        record = _selection_record(run_dir)
        if record["status"] != "final":
            continue
        stamp = record.get("evaluator_id", "")
        if stamp != head:
            excluded[run_dir.name] = (
                f"final under evaluator {stamp[:12] or '<unstamped>'}, "
                f"not head {head[:12]} — never re-ranked across rulers"
            )
            continue
        payload = rescore_run(snapshot, run_dir, env)
        score = float(payload["score"])
        scored[run_dir.name] = score
        if best is None or (
            score > scored[best] if higher_is_better else score < scored[best]
        ):
            best, best_payload = run_dir.name, payload

    metrics = {
        k: float(v)
        for k, v in (best_payload.get("metrics") or {}).items()
    }
    return FinalSelection(
        head_evaluator_id=head,
        winner_run=best,
        winner_score=scored.get(best) if best else None,
        winner_metrics=metrics,
        scored=scored,
        excluded=excluded,
    )
