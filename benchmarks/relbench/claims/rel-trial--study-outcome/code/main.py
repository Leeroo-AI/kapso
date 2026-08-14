# Imports

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from kapso_datasets.common import run_data_dir, shared_cache_dir
from publication_evidence import live_payload_probes
from registry_snapshot import ensure_snapshots


# Runtime

START = time.time()


def report(name: str, **values: object) -> None:
    payload = " ".join(f"{key}={value}" for key, value in values.items())
    print(f"[orchestrator] {name} elapsed={time.time() - START:.2f}s {payload}".rstrip(), flush=True)


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_stage(command: list[str], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["KAPSO_RUN_DATA_DIR"] = str(output)
    report("stage_start", command=" ".join(command), output=output)
    subprocess.run(command, check=True, env=environment)
    report("stage_complete", command=" ".join(command))


def save_incumbent(output: Path, name: str) -> Path:
    destination = shared_cache_dir() / "predictions" / name
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output / "val_predictions.npy", destination / "val_predictions.npy")
    shutil.copy2(output / "test_predictions.npy", destination / "test_predictions.npy")
    return destination


def build_publication_invariant() -> Path:
    cache = shared_cache_dir()
    source_path = cache / "predictions" / "generic_exp_0_core_channels_v1.npz"
    destination = cache / "predictions" / "generic_exp_0_invariant_channels_v1.npz"
    if destination.exists():
        return destination
    source = np.load(source_path, allow_pickle=False)
    train_size = len(source["train_nct_id"])
    available = []
    for raw_name in source["blend_names"]:
        name = str(raw_name)
        if f"oof_{name}" in source.files and f"val_{name}" in source.files and f"test_{name}" in source.files:
            values = np.asarray(source[f"oof_{name}"], dtype=np.float64)
            recent = pd.to_datetime(source["train_timestamp"]).year >= 2018
            if np.all(np.isfinite(values[recent])):
                available.append(name)
    if not available:
        available = [name for name in ["tabular", "word", "char", "structural"] if f"oof_{name}" in source.files]
    forward = np.full(train_size, np.nan, dtype=np.float64)
    forward[source["forward_index"].astype(int)] = source["forward_prediction"]
    recent = pd.to_datetime(source["train_timestamp"]).year >= 2018
    if np.all(np.isfinite(forward[recent])):
        available.append("external_compact")
    payload: dict[str, np.ndarray] = {
        "train_nct_id": source["train_nct_id"],
        "train_timestamp": source["train_timestamp"],
        "train_labels": np.asarray(load_train_labels(), dtype=np.int32),
        "validation_nct_id": load_split_ids("val"),
        "test_nct_id": load_split_ids("test"),
        "channel_names": np.asarray(available),
    }
    for name in available:
        if name == "external_compact":
            payload[f"oof_{name}"] = forward
            payload[f"validation_{name}"] = source["val"]
            payload[f"test_{name}"] = source["test"]
        else:
            payload[f"oof_{name}"] = source[f"oof_{name}"]
            payload[f"validation_{name}"] = source[f"val_{name}"]
            payload[f"test_{name}"] = source[f"test_{name}"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".npz.part")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    os.replace(temporary, destination)
    report("invariant", channels=",".join(available), path=destination)
    return destination


def load_train_labels() -> np.ndarray:
    from relbench.tasks import get_task

    task = get_task(os.environ["RELBENCH_DATASET"], os.environ["RELBENCH_TASK"], download=False)
    frame = task.get_table("train").df
    return frame[task.target_col].to_numpy(dtype=np.int32)


def load_split_ids(split: str) -> np.ndarray:
    from relbench.tasks import get_task

    task = get_task(os.environ["RELBENCH_DATASET"], os.environ["RELBENCH_TASK"], download=False)
    return task.get_table(split).df["nct_id"].to_numpy(dtype=np.int64)


def restore_candidate(debug: bool) -> bool:
    cache = shared_cache_dir()
    candidates = [
        cache / "predictions" / "generic_exp_4_uncapped_hosted_v1.npz",
        cache / "predictions" / "generic_exp_4_jats_direct_v1.npz",
        cache / "predictions" / "generic_exp_2_snapshot_direct_v1.npz",
    ]
    candidate = next((path for path in candidates if path.exists()), candidates[-1])
    if debug and not candidate.exists():
        candidate = cache / "predictions" / "generic_exp_2_snapshot_direct_debug_v1.npz"
    if not candidate.exists():
        return False
    artifact = np.load(candidate, allow_pickle=False)
    validation = np.asarray(artifact["val"], dtype=np.float64)
    test = np.asarray(artifact["test"], dtype=np.float64)
    if debug and "validation_routed" in artifact.files:
        archive_root = Path(os.environ["RELBENCH_WORK_DIR"]) / "runs" / "run_0005"
        baseline_validation = np.load(archive_root / "val_predictions.npy", allow_pickle=False).astype(np.float64)
        baseline_test = np.load(archive_root / "test_predictions.npy", allow_pickle=False).astype(np.float64)
        validation_indices = np.flatnonzero(artifact["validation_routed"].astype(bool))[:16]
        test_indices = np.flatnonzero(artifact["test_routed"].astype(bool))[:16]
        debug_validation = baseline_validation.copy()
        debug_test = baseline_test.copy()
        debug_validation[validation_indices] = validation[validation_indices]
        debug_test[test_indices] = test[test_indices]
        validation = debug_validation
        test = debug_test
    probes = live_payload_probes(cache, maximum_calls=2)
    output = run_data_dir()
    validation_path = output / "val_predictions.npy"
    np.save(validation_path, validation)
    frozen = checksum(validation_path)
    np.save(output / "test_predictions.npy", test)
    if checksum(validation_path) != frozen:
        raise RuntimeError("Model A validation checksum changed during cache restoration")
    diagnostics = json.loads(str(artifact["diagnostics_json"][0]))
    for stage_name, stage_path in {
        "medcpt_reranker": cache / "medcpt_endpoint_reranker_v1" / "diagnostics.json",
        "supplements": cache / "historical_supplements_v1" / "diagnostics.json",
    }.items():
        if stage_path.exists():
            diagnostics[stage_name] = json.loads(stage_path.read_text())
    diagnostics["official_live_probes"] = probes
    diagnostics["debug"] = debug
    diagnostics["debug_evidence_rows"] = 16 if debug else None
    diagnostics["validation_checksum"] = frozen
    (output / "metrics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True, allow_nan=True) + "\n")
    subprocess.run([sys.executable, "kapso_datasets/check_predictions.py"], check=True)
    report("cache_restore", candidate=candidate, live_probe_calls=probes["calls"], validation_checksum=frozen)
    return True


# Pipeline

def build(debug: bool) -> None:
    cache = shared_cache_dir()
    build_root = Path("output_data_generic_exp_4")
    if debug:
        run_stage([sys.executable, "jats_direct_pipeline.py", "--debug"], build_root / "jats_debug")
        run_stage([sys.executable, "hosted_widening_pipeline.py", "--debug"], build_root / "hosted_debug")
        if not restore_candidate(True):
            raise RuntimeError("The evidence-widening debug candidate could not be restored")
        return
    snapshot_root = cache / "registry_clock_lane0" / "projected"
    snapshot_diagnostics = ensure_snapshots(snapshot_root, cache / "registry_clock_lane0" / "downloads")
    report("snapshots", diagnostics=json.dumps(snapshot_diagnostics, sort_keys=True))
    run_stage([sys.executable, "direct_evidence_pipeline.py"], build_root / "direct")
    run_stage([sys.executable, "jats_direct_pipeline.py"], build_root / "jats")
    run_stage([sys.executable, "hosted_widening_pipeline.py"], build_root / "hosted")
    if restore_candidate(False):
        return
    core_output = build_root / "core"
    stage_marker = cache / "generic_exp_0_hosted_stage_complete.json"
    core_channels = cache / "predictions" / "generic_exp_0_core_channels_v1.npz"
    core_complete = all((core_output / name).exists() for name in ["val_predictions.npy", "test_predictions.npy", "metrics.json"]) and core_channels.exists()
    if not core_complete:
        if not stage_marker.exists():
            run_stage([sys.executable, "champion_core.py"], build_root / "core_hosted_warmup")
        run_stage([sys.executable, "champion_core.py"], core_output)
    else:
        report("core_cache", state="hit", validation_checksum=checksum(core_output / "val_predictions.npy"))
    save_incumbent(core_output, "generic_exp_0_incumbent_core")
    build_publication_invariant()
    registry_output = build_root / "registry"
    run_stage([sys.executable, "registry_pipeline.py"], registry_output)
    save_incumbent(registry_output, "generic_exp_0_incumbent_registry")
    publication_output = build_root / "literature_v3"
    run_stage([sys.executable, "publication_pipeline.py"], publication_output)
    source = cache / "predictions" / "generic_exp_0_literature_v3.npz"
    if not source.exists():
        raise RuntimeError("The literature v3 stage did not persist its candidate")
    if not restore_candidate(False):
        raise RuntimeError("The persisted literature v3 candidate could not be restored")


def main() -> None:
    debug = "--debug" in sys.argv
    report("start", debug=debug)
    if not debug and restore_candidate(False):
        return
    build(debug)


if __name__ == "__main__":
    main()
