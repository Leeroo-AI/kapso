# Trajectory store tests — the P1 contract (docs/plans/learning/p1-trajectory-store.md).
#
# Each test names the regression it catches (Rule 9). All hermetic: local-only
# stores under tmp_path, no GCS, no network.

import json
import shutil
import tarfile

import pytest
import yaml

from kapso.learning.corpus_import import (
    import_archive,
    import_subset,
    trajectory_id_from_archive_uri,
)
from kapso.learning.trajectory_store import (
    MANIFEST_NAME,
    TrajectoryStore,
    save_trajectory,
    validate_trajectory_id,
)

TRAJECTORY_ID = "rel-amazon--user-churn/20260101T000000_lane-t1"

FINAL_REPORT = {
    "dataset": "rel-amazon",
    "task": "user-churn",
    "family": "entity_binary_classification",
    "primary_metric": "roc_auc",
    "run": "run_0001",
    "val_metrics": {"roc_auc": 0.7136, "f1": 0.68},
    "test_metrics": {"roc_auc": 0.7155},
}


def build_work_dir(root, strict=True):
    """A minimal campaign work dir; strict=False mimics a historical archive
    (no living documents, no workspace .kapso files)."""
    work_dir = root / "work"
    (work_dir / "runs" / "run_0001").mkdir(parents=True)
    (work_dir / "runs" / "run_0001" / "metrics.json").write_text('{"roc_auc": 0.7136}')
    (work_dir / "campaign_meta.json").write_text('{"lane": "lane-t1", "hardware": "cpu"}')
    (work_dir / "final_report.json").write_text(json.dumps(FINAL_REPORT))
    (work_dir / "__pycache__").mkdir()
    (work_dir / "__pycache__" / "junk.cpython-312.pyc").write_bytes(b"\x00")
    if strict:
        (work_dir / "features_history.md").write_text("## F1 — KEPT (+0.003)\n")
        (work_dir / "lens_plan_history.jsonl").write_text('{"iteration": 1}\n')
        (work_dir / "experiment_history.json").write_text("[]")
    log = root / "campaign_src.log"
    log.write_text("campaign started\n")
    return work_dir, log


def make_store(tmp_path):
    return TrajectoryStore(local=str(tmp_path / "store"))


def save(store, root, contract="strict", trajectory_id=TRAJECTORY_ID):
    work_dir, log = build_work_dir(root, strict=(contract == "strict"))
    return save_trajectory(
        store, trajectory_id, work_dir=str(work_dir), campaign_log=str(log),
        contract=contract,
    )


def test_save_load_roundtrip(tmp_path):
    # Regression: the three read doors must serve exactly what save gathered,
    # with the design's manifest fields derived correctly.
    store = make_store(tmp_path)
    save(store, tmp_path)
    manifest = store.manifest(TRAJECTORY_ID)
    assert manifest["id"] == TRAJECTORY_ID
    assert manifest["task"] == "rel-amazon/user-churn"
    assert manifest["created"] == "2026-01-01T00:00:00Z"
    assert manifest["contract"] == "strict"
    assert manifest["family"] == "entity_binary_classification"
    assert manifest["outcome"] == {
        "selected_run": "run_0001",
        "val": {"roc_auc": 0.7136},
        "test": {"roc_auc": 0.7155},
    }
    assert "missing" not in manifest
    bundle_dir = store.resolve(TRAJECTORY_ID)
    assert (bundle_dir / "campaign.log").read_text() == "campaign started\n"
    ref = store.open_ref(TRAJECTORY_ID, "runs/run_0001/metrics.json#anchor")
    assert json.loads(ref.read_text())["roc_auc"] == 0.7136
    run_dir = store.resolve(TRAJECTORY_ID, "runs/run_0001")
    assert (run_dir / "metrics.json").is_file()


def test_gather_excludes_pycache(tmp_path):
    # Regression: derived caches are noise — a .pyc must never enter the
    # inventory (mining hashes every listed file against the manifest).
    store = make_store(tmp_path)
    save(store, tmp_path)
    inventory = store.manifest(TRAJECTORY_ID)["inventory"]
    assert not [p for p in inventory["sha256"] if "__pycache__" in p]
    assert inventory["files"] == len(inventory["sha256"])


def test_double_save_is_idempotent(tmp_path):
    # Regression: re-importing the corpus must be a no-op, never a duplicate
    # or a clobber.
    store = make_store(tmp_path)
    save(store, tmp_path / "a")
    first = store.manifest(TRAJECTORY_ID)
    save(store, tmp_path / "b")
    assert store.manifest(TRAJECTORY_ID)["inventory"] == first["inventory"]


def test_mismatched_resave_raises(tmp_path):
    # Regression: an existing id with different content must raise, never be
    # silently overwritten (design §3.4 register semantics).
    store = make_store(tmp_path)
    save(store, tmp_path / "a")
    root_b = tmp_path / "b"
    work_dir, log = build_work_dir(root_b)
    (work_dir / "runs" / "run_0001" / "metrics.json").write_text('{"roc_auc": 0.9}')
    with pytest.raises(FileExistsError):
        save_trajectory(
            store, TRAJECTORY_ID, work_dir=str(work_dir), campaign_log=str(log)
        )


def test_manifestless_bundle_is_invisible(tmp_path):
    # Regression: manifest-last is the commit marker — a prefix without
    # trajectory.yaml (torn write / partial upload) must not resolve.
    store = make_store(tmp_path)
    save(store, tmp_path)
    (store.local / TRAJECTORY_ID / MANIFEST_NAME).unlink()
    with pytest.raises(FileNotFoundError):
        store.manifest(TRAJECTORY_ID)
    with pytest.raises(FileNotFoundError):
        store.resolve(TRAJECTORY_ID)


def test_corrupt_manifest_raises(tmp_path):
    # Regression: a corrupt manifest must raise (Rule 2), not parse to junk.
    store = make_store(tmp_path)
    save(store, tmp_path)
    manifest_path = store.local / TRAJECTORY_ID / MANIFEST_NAME
    manifest_path.write_text("id: [unclosed")
    with pytest.raises(yaml.YAMLError):
        store.manifest(TRAJECTORY_ID)
    manifest_path.write_text("id: some-other/20260101T000000_lane-x\ninventory: {}")
    with pytest.raises(ValueError):
        store.manifest(TRAJECTORY_ID)


def test_strict_contract_rejects_thin_bundle(tmp_path):
    # Regression: the strict completeness contract — a new harvest missing the
    # living documents is not a trajectory (no thin saves).
    store = make_store(tmp_path)
    work_dir, log = build_work_dir(tmp_path, strict=False)
    with pytest.raises(FileNotFoundError, match="features_history.md"):
        save_trajectory(
            store, TRAJECTORY_ID, work_dir=str(work_dir), campaign_log=str(log)
        )


def test_historical_contract_records_missing(tmp_path):
    # Regression: the historical relaxation is explicit and visible — the
    # manifest must record exactly which strict parts are absent.
    store = make_store(tmp_path)
    save(store, tmp_path, contract="historical")
    manifest = store.manifest(TRAJECTORY_ID)
    assert manifest["contract"] == "historical"
    assert sorted(manifest["missing"]) == [
        "experiment_history.json",
        "features_history.md",
        "lens_plan_history.jsonl",
    ]


def test_open_ref_is_the_only_door(tmp_path):
    # Regression: refs outside the bundle (escape) or outside the inventory
    # (unknown) must raise — cards only ever cite store-resolvable refs.
    store = make_store(tmp_path)
    save(store, tmp_path)
    with pytest.raises(ValueError):
        store.open_ref(TRAJECTORY_ID, "../outside.txt")
    with pytest.raises(KeyError):
        store.open_ref(TRAJECTORY_ID, "runs/run_0001/nonexistent.json")
    with pytest.raises(KeyError):
        store.resolve(TRAJECTORY_ID, "no/such/prefix")


def test_invalid_trajectory_id_rejected(tmp_path):
    # Regression: the id is the path in both stores — malformed ids must never
    # reach the filesystem layer.
    with pytest.raises(ValueError):
        validate_trajectory_id("no-slash-here")
    with pytest.raises(ValueError):
        validate_trajectory_id("a/b/c")
    store = make_store(tmp_path)
    with pytest.raises(ValueError):
        store.manifest("../../etc/passwd")


def build_archive(tmp_path, task_dir="rel-amazon--user-churn",
                  stamp_lane="20260101T000000_lane-t1", break_it=False):
    """A historical archive tarball in the real layout:
    tmp/relbench/<task-dir>/ + tmp/campaign_<task-dir>.log."""
    root = tmp_path / "archive_src" / task_dir / stamp_lane
    payload = root / "tmp" / "relbench" / task_dir
    (payload / "runs" / "run_0001").mkdir(parents=True)
    (payload / "runs" / "run_0001" / "metrics.json").write_text('{"roc_auc": 0.7136}')
    (payload / "campaign_meta.json").write_text('{"lane": "t1"}')
    if not break_it:
        (payload / "final_report.json").write_text(json.dumps(FINAL_REPORT))
    (root / "tmp" / f"campaign_{task_dir}.log").write_text("log line\n")
    archive_dir = tmp_path / "archives" / task_dir
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"{stamp_lane}.tgz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(root / "tmp", arcname="tmp")
    shutil.rmtree(root)
    return archive_path


def test_import_archive_end_to_end(tmp_path):
    # Regression: the historical import gather — work dir at bundle root,
    # campaign.log named per layout, historical contract recorded, id derived
    # from the archive path.
    store = make_store(tmp_path)
    archive_path = build_archive(tmp_path)
    derived = trajectory_id_from_archive_uri(str(archive_path))
    assert derived == TRAJECTORY_ID
    outcome = import_archive(store, str(archive_path))
    assert outcome == {"id": TRAJECTORY_ID, "status": "imported"}
    manifest = store.manifest(TRAJECTORY_ID)
    assert manifest["contract"] == "historical"
    assert (store.resolve(TRAJECTORY_ID) / "campaign.log").read_text() == "log line\n"
    # scratch space cleaned
    assert not any((store.local / "_incoming").iterdir())
    # re-import is a no-op
    assert import_archive(store, str(archive_path))["status"] == "already-registered"


def test_import_subset_reports_and_raises_on_validation_failure(tmp_path):
    # Regression: a contract-violating archive becomes a named finding in the
    # report and a loud final raise — never a silent skip, never stranding the
    # healthy entries behind it.
    store = make_store(tmp_path)
    good = build_archive(tmp_path)
    bad = build_archive(
        tmp_path, task_dir="rel-hm--user-churn",
        stamp_lane="20260102T000000_lane-t2", break_it=True,
    )
    subset_path = tmp_path / "subset.yaml"
    subset_path.write_text(yaml.safe_dump({"trajectories": [
        {"id": TRAJECTORY_ID, "archive": str(good), "role": "learn"},
        {"id": "rel-hm--user-churn/20260102T000000_lane-t2",
         "archive": str(bad), "role": "learn"},
    ]}))
    report_dir = tmp_path / "reports"
    with pytest.raises(RuntimeError, match="1 subset entry failed"):
        import_subset(store, str(subset_path), str(report_dir))
    report_text = next(report_dir.glob("import-*.md")).read_text()
    assert "failed-validation — work dir missing required final_report.json" in report_text
    # the healthy entry still landed
    assert store.manifest(TRAJECTORY_ID)["contract"] == "historical"


def test_store_from_config_requires_block(tmp_path):
    # Regression: config is the single source (Rule 1) — a config without the
    # learning block must fail loud, not fall back to a buried default.
    with pytest.raises(KeyError):
        TrajectoryStore.from_config({})
    store = TrajectoryStore.from_config(
        {"learning": {"trajectory_store": {"local": str(tmp_path / "s"), "remote": None}}}
    )
    assert store.remote is None
