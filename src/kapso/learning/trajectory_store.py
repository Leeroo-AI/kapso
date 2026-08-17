# Trajectory store — identity-addressed campaign bundles.
#
# Design: learn-from-trajectories-design.md §3.4. The store is object-storage
# prefixes plus a cache-through local mirror: reference by identity, resolve by
# store. Bundles are stored unpacked (one object per file), never as tarballs;
# atomicity is recovered by writing the manifest last as the commit marker.
# Reads go through exactly three doors — manifest / resolve / open_ref — and
# there is no other door.

import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

MANIFEST_NAME = "trajectory.yaml"
CAMPAIGN_LOG_NAME = "campaign.log"

# Completeness contract (design §3.4). `strict` governs new harvests; a
# trajectory that cannot support ledger refs is not a trajectory — no thin
# saves. `historical` is the explicit relaxation for archives that predate the
# store (the wave-4 forensics gap: workspace .kapso files and living documents
# were never archived); the manifest records exactly what is missing, so
# thin-ness is visible to mining, never silent.
REQUIRED_FILES_STRICT = (
    "campaign_meta.json",
    "final_report.json",
    CAMPAIGN_LOG_NAME,
    "features_history.md",
    "lens_plan_history.jsonl",
    "experiment_history.json",
)
REQUIRED_FILES_HISTORICAL = (
    "campaign_meta.json",
    "final_report.json",
    CAMPAIGN_LOG_NAME,
)
CONTRACTS = ("strict", "historical")

# Derived caches are noise, not evidence — excluded at gather time.
GATHER_EXCLUDED_DIR_NAMES = ("__pycache__",)

# <task-dir>/<stamp>_<lane>, e.g. rel-amazon--user-churn/20260813T015420_lane-c10
TRAJECTORY_ID_PATTERN = re.compile(
    r"^(?P<task>[A-Za-z0-9._-]+)/(?P<stamp>\d{8}T\d{6})_(?P<lane>[A-Za-z0-9-]+)$"
)


def validate_trajectory_id(trajectory_id: str) -> Dict[str, str]:
    """Validate a trajectory id and return its parsed parts (task/stamp/lane)."""
    match = TRAJECTORY_ID_PATTERN.match(trajectory_id)
    if match is None:
        raise ValueError(
            f"invalid trajectory id {trajectory_id!r}: expected "
            "'<task-dir>/<YYYYMMDDTHHMMSS>_<lane>'"
        )
    return match.groupdict()


def _clean_ref(ref: str) -> str:
    """Normalize an evidence ref to a bundle-relative path (fragment stripped)."""
    path_part = ref.split("#", 1)[0]
    if not path_part:
        raise ValueError(f"ref {ref!r} has no path component")
    if path_part.startswith("/") or ".." in Path(path_part).parts:
        raise ValueError(f"ref {ref!r} escapes the bundle root")
    return path_part


def _hash_file(path: Path) -> str:
    digest = sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory_of(bundle_dir: Path) -> Dict[str, Any]:
    """Hash every file under bundle_dir (manifest excluded) into an inventory."""
    hashes: Dict[str, str] = {}
    total_bytes = 0
    for path in sorted(bundle_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(bundle_dir).as_posix()
        if rel == MANIFEST_NAME:
            continue
        hashes[rel] = _hash_file(path)
        total_bytes += path.stat().st_size
    return {"files": len(hashes), "bytes": total_bytes, "sha256": hashes}


def _created_from_stamp(stamp: str) -> str:
    created = datetime.strptime(stamp, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    return created.isoformat().replace("+00:00", "Z")


def _load_final_report(bundle_dir: Path) -> Dict[str, Any]:
    """Parse final_report.json; a malformed report raises (fail loud)."""
    with open(bundle_dir / "final_report.json") as handle:
        report = json.load(handle)
    if not isinstance(report, dict):
        raise ValueError(f"final_report.json in {bundle_dir} is not a mapping")
    return report


def _outcome_from_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the manifest outcome block from a parsed final report.

    Fields are present-when-derivable: historical reports carry no cost or
    iteration counts, and omitting a key is honest where fabricating one is
    not.
    """
    outcome: Dict[str, Any] = {}
    if "run" in report:
        outcome["selected_run"] = report["run"]
    metric = report.get("primary_metric")
    for side in ("val", "test"):
        metrics = report.get(f"{side}_metrics")
        if metric and isinstance(metrics, dict) and metric in metrics:
            outcome[side] = {metric: metrics[metric]}
    return outcome


def _run_dirs(bundle_dir: Path) -> List[Path]:
    runs_dir = bundle_dir / "runs"
    if not runs_dir.is_dir():
        return []
    return sorted(p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("run_"))


def _validate_contract(bundle_dir: Path, contract: str) -> List[str]:
    """Check the completeness contract; return the strict-parts missing list.

    Under `strict` any missing part raises. Under `historical` the core parts
    (meta, report, log, >=1 run) still raise when absent, and the remaining
    strict parts are returned so the manifest can record them as missing.
    """
    if contract not in CONTRACTS:
        raise ValueError(f"unknown contract {contract!r}: expected one of {CONTRACTS}")
    required = REQUIRED_FILES_STRICT if contract == "strict" else REQUIRED_FILES_HISTORICAL
    absent_required = [name for name in required if not (bundle_dir / name).is_file()]
    if absent_required:
        raise FileNotFoundError(
            f"bundle at {bundle_dir} violates the {contract} completeness "
            f"contract: missing {absent_required}"
        )
    if not _run_dirs(bundle_dir):
        raise FileNotFoundError(
            f"bundle at {bundle_dir} violates the {contract} completeness "
            "contract: runs/ has no registered run"
        )
    return [
        name for name in REQUIRED_FILES_STRICT
        if name not in required and not (bundle_dir / name).is_file()
    ]


class TrajectoryStore:
    """The trajectory store: a local mirror, optionally backed by a remote.

    No remote configured -> the store is the local directory and everything
    still resolves (local-only users). With a remote, reads are cache-through:
    `manifest` is one small GET, `open_ref` is a single-object GET, `resolve`
    materializes a prefix.
    """

    def __init__(self, local: str, remote: Optional[str] = None):
        if not local:
            raise ValueError("trajectory store requires a local directory")
        self.local = Path(local).expanduser()
        self.remote = remote.rstrip("/") if remote else None
        self.local.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TrajectoryStore":
        """Build from a loaded config dict (`learning.trajectory_store`, Rule 1)."""
        learning = config.get("learning")
        if not isinstance(learning, dict):
            raise KeyError("config has no `learning:` block")
        block = learning.get("trajectory_store")
        if not isinstance(block, dict) or "local" not in block:
            raise KeyError("config `learning.trajectory_store` must define `local`")
        return cls(local=block["local"], remote=block.get("remote"))

    # ------------------------------------------------------------------ paths

    def _local_dir(self, trajectory_id: str) -> Path:
        validate_trajectory_id(trajectory_id)
        return self.local / trajectory_id

    def _remote_prefix(self, trajectory_id: str) -> str:
        if self.remote is None:
            raise FileNotFoundError(
                f"trajectory {trajectory_id!r} is not resident locally and no "
                "remote store is configured"
            )
        return f"{self.remote}/{trajectory_id}"

    def _fetch_file(self, trajectory_id: str, rel_path: str) -> Path:
        """Single-object GET of one bundle file into the local mirror."""
        target = self._local_dir(trajectory_id) / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        source = f"{self._remote_prefix(trajectory_id)}/{rel_path}"
        subprocess.run(["gsutil", "cp", source, str(target)], check=True)
        return target

    # ------------------------------------------------------------- read doors

    def list_manifests(self) -> List[Dict[str, Any]]:
        """Parsed manifests of every locally resident trajectory.

        Enumeration is local-mirror only (batch drivers operate on what has
        been materialized); remote-wide listing is not a v1 need.
        """
        manifests = []
        for manifest_path in sorted(self.local.glob(f"*/*/{MANIFEST_NAME}")):
            trajectory_id = manifest_path.parent.relative_to(self.local).as_posix()
            if TRAJECTORY_ID_PATTERN.match(trajectory_id):
                manifests.append(self.manifest(trajectory_id))
        return manifests

    def manifest(self, trajectory_id: str) -> Dict[str, Any]:
        """Parsed trajectory.yaml; one GET when the bundle is not resident."""
        local_manifest = self._local_dir(trajectory_id) / MANIFEST_NAME
        if not local_manifest.is_file():
            self._fetch_file(trajectory_id, MANIFEST_NAME)
        with open(local_manifest) as handle:
            manifest = yaml.safe_load(handle)
        if not isinstance(manifest, dict) or manifest.get("id") != trajectory_id:
            raise ValueError(
                f"corrupt manifest for {trajectory_id!r}: id mismatch or not a mapping"
            )
        return manifest

    def resolve(self, trajectory_id: str, subpath: Optional[str] = None) -> Path:
        """Local path for the bundle (or a subpath), cache-through.

        Materialization is inventory-driven: every manifest-listed file under
        the requested prefix must exist locally, and missing ones are pulled.
        Existence, not re-hashing, is the residency check — integrity auditing
        against the manifest hashes is its consumers' job (mining's raw-
        immutability check, import verification).
        """
        manifest = self.manifest(trajectory_id)
        prefix = _clean_ref(subpath) if subpath is not None else None
        wanted = [
            rel for rel in manifest["inventory"]["sha256"]
            if prefix is None or rel == prefix or rel.startswith(prefix + "/")
        ]
        if prefix is not None and not wanted:
            raise KeyError(
                f"subpath {subpath!r} matches no file in {trajectory_id!r}'s inventory"
            )
        bundle_dir = self._local_dir(trajectory_id)
        for rel in wanted:
            if not (bundle_dir / rel).is_file():
                self._fetch_file(trajectory_id, rel)
        return bundle_dir if prefix is None else bundle_dir / prefix

    def open_ref(self, trajectory_id: str, ref: str) -> Path:
        """Local path for one evidence ref; a single GET when non-resident."""
        rel = _clean_ref(ref)
        manifest = self.manifest(trajectory_id)
        if rel not in manifest["inventory"]["sha256"]:
            raise KeyError(f"ref {ref!r} is not in {trajectory_id!r}'s inventory")
        target = self._local_dir(trajectory_id) / rel
        if not target.is_file():
            self._fetch_file(trajectory_id, rel)
        return target

    # ------------------------------------------------------------------ write

    def register(self, staged_dir: Path, trajectory_id: str, upload: Optional[bool]) -> str:
        """Finalize a staged bundle: idempotent local commit, then upload.

        An existing id with matching inventory is a no-op; mismatching content
        raises (never silently overwritten). The staged dir must already
        contain its manifest (written last by save_trajectory); the local
        commit is one atomic rename.
        """
        final_dir = self._local_dir(trajectory_id)
        with open(staged_dir / MANIFEST_NAME) as handle:
            staged_manifest = yaml.safe_load(handle)
        if final_dir.exists():
            existing = self.manifest(trajectory_id)
            if existing["inventory"]["sha256"] != staged_manifest["inventory"]["sha256"]:
                raise FileExistsError(
                    f"trajectory {trajectory_id!r} already registered with "
                    "different content; refusing to overwrite"
                )
            shutil.rmtree(staged_dir)
        else:
            final_dir.parent.mkdir(parents=True, exist_ok=True)
            staged_dir.rename(final_dir)
        if upload is None:
            upload = self.remote is not None
        if upload:
            self._upload(trajectory_id)
        return trajectory_id

    def upload_derived(self, trajectory_id: str, subdir: str) -> None:
        """Mirror a derived layer (e.g. mined/) plus the updated manifest."""
        bundle_dir = self._local_dir(trajectory_id)
        prefix = self._remote_prefix(trajectory_id)
        subprocess.run(
            ["gsutil", "-m", "rsync", "-r", str(bundle_dir / subdir),
             f"{prefix}/{subdir}"],
            check=True,
        )
        subprocess.run(
            ["gsutil", "cp", str(bundle_dir / MANIFEST_NAME),
             f"{prefix}/{MANIFEST_NAME}"],
            check=True,
        )

    def _upload(self, trajectory_id: str) -> None:
        """Upload the unpacked prefix, manifest last (the remote commit marker)."""
        bundle_dir = self._local_dir(trajectory_id)
        prefix = self._remote_prefix(trajectory_id)
        subprocess.run(
            ["gsutil", "-m", "rsync", "-r", "-x", r"^trajectory\.yaml$",
             str(bundle_dir), prefix],
            check=True,
        )
        subprocess.run(
            ["gsutil", "cp", str(bundle_dir / MANIFEST_NAME), f"{prefix}/{MANIFEST_NAME}"],
            check=True,
        )


def _gather(
    staged_dir: Path,
    work_dir: Path,
    campaign_log: Path,
    extra_files: Optional[Dict[str, Path]],
    work_dir_exclude: tuple,
) -> None:
    """Assemble the bundle layout by gathering, never renaming (design §3.4).

    Work-dir contents land at the bundle root so existing ref habits
    (`runs/run_0019/...`, `features_history.md#anchor`) stay valid. The one
    layout-specified name is `campaign.log`. `extra_files` maps bundle-relative
    names to sources — files or directories (the workspace .kapso files,
    living documents, and session/ideation dirs that strict harvests supply).
    `work_dir_exclude` names top-level work-dir entries that are not campaign
    evidence (e.g. the shared cache with its model caches).
    """
    if not work_dir.is_dir():
        raise FileNotFoundError(f"work_dir {work_dir} is not a directory")
    if not campaign_log.is_file():
        raise FileNotFoundError(f"campaign_log {campaign_log} is not a file")

    def _ignore(directory: str, names: List[str]) -> List[str]:
        ignored = [n for n in names if n in GATHER_EXCLUDED_DIR_NAMES]
        if Path(directory) == work_dir:
            ignored += [n for n in names if n in work_dir_exclude]
        return ignored

    shutil.copytree(work_dir, staged_dir, ignore=_ignore, dirs_exist_ok=True)
    shutil.copy2(campaign_log, staged_dir / CAMPAIGN_LOG_NAME)
    for rel_name, source in (extra_files or {}).items():
        rel = _clean_ref(rel_name)
        target = staged_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(
                source, target,
                ignore=shutil.ignore_patterns(*GATHER_EXCLUDED_DIR_NAMES),
            )
        elif source.is_file():
            shutil.copy2(source, target)
        else:
            raise FileNotFoundError(f"extra item {source} for {rel_name!r} is missing")


def save_trajectory(
    store: TrajectoryStore,
    trajectory_id: str,
    work_dir: str,
    campaign_log: str,
    extra_files: Optional[Dict[str, str]] = None,
    work_dir_exclude: tuple = (),
    contract: str = "strict",
    kapso_commit: Optional[str] = None,
    bank_head: Optional[str] = None,
    upload: Optional[bool] = None,
) -> str:
    """Gather -> validate -> hash -> register (idempotent) -> upload (§3.4).

    This is the harvest step and the evolve->learn bridge. `contract` is
    `strict` for new harvests; the corpus importer passes `historical`
    explicitly for pre-store archives, and the manifest then records the
    missing strict parts. Returns the trajectory id.
    """
    parts = validate_trajectory_id(trajectory_id)
    staged_dir = store.local / "_staging" / trajectory_id.replace("/", "--")
    if staged_dir.exists():
        shutil.rmtree(staged_dir)
    staged_dir.mkdir(parents=True)

    _gather(
        staged_dir,
        Path(work_dir).expanduser(),
        Path(campaign_log).expanduser(),
        {name: Path(path).expanduser() for name, path in (extra_files or {}).items()},
        work_dir_exclude,
    )
    missing = _validate_contract(staged_dir, contract)
    report = _load_final_report(staged_dir)

    manifest: Dict[str, Any] = {
        "id": trajectory_id,
        "task": parts["task"].replace("--", "/"),
        "created": _created_from_stamp(parts["stamp"]),
        "kapso_commit": kapso_commit,
        "bank_head": bank_head,
        "contract": contract,
        "dataset": report.get("dataset"),
        "family": report.get("family"),
        "outcome": _outcome_from_report(report),
        "inventory": _inventory_of(staged_dir),
    }
    if missing:
        manifest["missing"] = missing

    # Manifest written last — the commit marker.
    with open(staged_dir / MANIFEST_NAME, "w") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False)

    return store.register(staged_dir, trajectory_id, upload)
