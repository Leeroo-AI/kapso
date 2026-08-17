# Corpus import — historical campaign archives into the trajectory store.
#
# Design: learn-from-trajectories-design.md §3.4 + plan p1-trajectory-store.md.
# Historical archives (`gs://.../runs/<task>/<stamp>_<lane>.tgz`) predate the
# store: a tarball holds `tmp/relbench/<task-dir>/` (the campaign work dir) and
# `tmp/campaign_<task-dir>.log`. Import = download -> unpack -> validate ->
# gather -> save_trajectory under the explicit `historical` contract (the
# manifest then records the strict parts these archives never carried).
#
# Failure discipline (Rule 2, no try/except): contract violations are found by
# explicit checks and recorded as named findings in the report — the run
# continues past them and raises at the end. Genuine corruption (an unreadable
# tarball, a failed download, a malformed JSON) raises immediately and stops
# the run loudly; import is idempotent, so a stopped run re-runs safely.

import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from kapso.learning.trajectory_store import (
    REQUIRED_FILES_HISTORICAL,
    TrajectoryStore,
    save_trajectory,
    validate_trajectory_id,
)


def trajectory_id_from_archive_uri(archive_uri: str) -> str:
    """Derive `<task-dir>/<stamp>_<lane>` from an archive path or URI."""
    path = Path(archive_uri)
    if path.suffix != ".tgz":
        raise ValueError(f"archive {archive_uri!r} is not a .tgz")
    trajectory_id = f"{path.parent.name}/{path.stem}"
    validate_trajectory_id(trajectory_id)
    return trajectory_id


def _fetch_archive(archive_uri: str, scratch_dir: Path) -> Path:
    """Materialize the archive locally (gsutil for gs:// URIs)."""
    target = scratch_dir / Path(archive_uri).name
    if archive_uri.startswith("gs://"):
        subprocess.run(["gsutil", "cp", archive_uri, str(target)], check=True)
        return target
    source = Path(archive_uri).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"archive {archive_uri!r} does not exist")
    shutil.copy2(source, target)
    return target


def _validate_archive_layout(extract_dir: Path) -> Dict[str, Any]:
    """Explicit layout + historical-contract checks on an unpacked archive.

    Returns {work_dir, campaign_log, findings}; a non-empty findings list
    means the archive fails validation (recorded, never silently skipped).
    """
    findings: List[str] = []
    work_dirs = [p for p in extract_dir.glob("tmp/*/*") if p.is_dir()]
    logs = list(extract_dir.glob("tmp/campaign_*.log"))
    if len(work_dirs) != 1:
        findings.append(
            f"expected exactly one work dir under tmp/<benchmark>/<task>, "
            f"found {len(work_dirs)}"
        )
    if len(logs) != 1:
        findings.append(f"expected exactly one tmp/campaign_*.log, found {len(logs)}")
    work_dir = work_dirs[0] if len(work_dirs) == 1 else None
    if work_dir is not None:
        for name in REQUIRED_FILES_HISTORICAL:
            if name != "campaign.log" and not (work_dir / name).is_file():
                findings.append(f"work dir missing required {name}")
        runs_dir = work_dir / "runs"
        run_dirs = (
            [p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
            if runs_dir.is_dir()
            else []
        )
        if not run_dirs:
            findings.append("runs/ has no registered run")
    return {
        "work_dir": work_dir,
        "campaign_log": logs[0] if len(logs) == 1 else None,
        "findings": findings,
    }


def import_archive(
    store: TrajectoryStore,
    archive_uri: str,
    trajectory_id: Optional[str] = None,
    upload: Optional[bool] = None,
) -> Dict[str, str]:
    """Import one historical archive.

    Returns {id, status}: `imported`, `already-registered`, or
    `failed-validation — <findings>`.
    """
    trajectory_id = trajectory_id or trajectory_id_from_archive_uri(archive_uri)
    validate_trajectory_id(trajectory_id)
    if (store.local / trajectory_id / "trajectory.yaml").is_file():
        return {"id": trajectory_id, "status": "already-registered"}

    scratch_dir = store.local / "_incoming" / trajectory_id.replace("/", "--")
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)
    scratch_dir.mkdir(parents=True)

    archive_path = _fetch_archive(archive_uri, scratch_dir)
    extract_dir = scratch_dir / "extracted"
    extract_dir.mkdir()
    with tarfile.open(archive_path) as archive:
        archive.extractall(extract_dir, filter="data")

    layout = _validate_archive_layout(extract_dir)
    if layout["findings"]:
        shutil.rmtree(scratch_dir)
        return {
            "id": trajectory_id,
            "status": "failed-validation — " + "; ".join(layout["findings"]),
        }

    save_trajectory(
        store,
        trajectory_id,
        work_dir=str(layout["work_dir"]),
        campaign_log=str(layout["campaign_log"]),
        contract="historical",
        upload=upload,
    )
    shutil.rmtree(scratch_dir)
    return {"id": trajectory_id, "status": "imported"}


def load_subset(subset_path: str) -> List[Dict[str, Any]]:
    """Parse a subset file: a `trajectories:` list of {id, archive, role}."""
    with open(subset_path) as handle:
        subset = yaml.safe_load(handle)
    entries = subset.get("trajectories") if isinstance(subset, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"subset file {subset_path!r} has no `trajectories:` list")
    for entry in entries:
        if not isinstance(entry, dict) or "archive" not in entry or "id" not in entry:
            raise ValueError(f"subset entry {entry!r} must carry `id` and `archive`")
        validate_trajectory_id(entry["id"])
    return entries


def import_subset(
    store: TrajectoryStore,
    subset_path: str,
    report_dir: str,
    upload: Optional[bool] = None,
) -> Path:
    """Import every subset entry; write the import report; raise if any failed.

    Contract-violating archives are recorded as named findings and the run
    continues past them (one thin archive does not strand the corpus); the
    final raise makes the failure loud. Genuine corruption stops the run
    immediately (import_archive raises), and a re-run is safe by idempotency.
    """
    entries = load_subset(subset_path)
    results = [
        {
            "id": entry["id"],
            "role": entry.get("role", ""),
            **import_archive(store, entry["archive"], entry["id"], upload),
        }
        for entry in entries
    ]
    failures = sum(1 for r in results if r["status"].startswith("failed-validation"))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    report_path = Path(report_dir).expanduser() / f"import-{stamp}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Trajectory corpus import report",
        "",
        f"- subset: `{subset_path}`",
        f"- store: `{store.local}` (remote: `{store.remote}`)",
        f"- imported: {sum(1 for r in results if r['status'] == 'imported')}"
        f" · already-registered: "
        f"{sum(1 for r in results if r['status'] == 'already-registered')}"
        f" · failed-validation: {failures}",
        "",
        "| id | role | status |",
        "|---|---|---|",
    ]
    lines += [f"| {r['id']} | {r['role']} | {r['status']} |" for r in results]
    report_path.write_text("\n".join(lines) + "\n")
    if failures:
        raise RuntimeError(
            f"{failures} subset entr{'y' if failures == 1 else 'ies'} failed "
            f"validation; findings in {report_path}"
        )
    return report_path
