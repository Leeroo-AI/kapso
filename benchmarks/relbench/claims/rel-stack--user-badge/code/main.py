from __future__ import annotations

import json
import random
import sys
import time
import warnings

import numpy as np

from cold_replay import build_panel, cache_root, final_predictions, register_artifact, run_gate, verify_fingerprints
from kapso_datasets.common import is_debug, load_task, run_data_dir


# Entrypoint

def main() -> None:
    warnings.filterwarnings("ignore")
    np.random.seed(1337)
    random.seed(1337)
    debug = is_debug()
    started = time.time()
    ctx = load_task()
    train = ctx.train.df.reset_index(drop=True)
    validation = ctx.val.df.reset_index(drop=True)
    test = ctx.test.df.reset_index(drop=True)
    fingerprints = verify_fingerprints(train, validation, test)
    print(f"[runtime] phase=fingerprints elapsed_seconds={time.time() - started:.1f}", flush=True)
    frozen = cache_root() / "frozen_candidate.json"
    if not debug and frozen.exists():
        validation_prediction = np.load(cache_root() / "candidate_val.npy").astype(np.float32)
        test_prediction = np.load(cache_root() / "candidate_test.npy").astype(np.float32)
        diagnostics = json.loads(frozen.read_text())
        diagnostics["cache_reused"] = True
        output = run_data_dir()
        np.save(output / "val_predictions.npy", validation_prediction)
        np.save(output / "test_predictions.npy", test_prediction)
        (output / "metrics.json").write_text(json.dumps(diagnostics, indent=2))
        print(f"[diagnostics] {json.dumps(diagnostics,separators=(',', ':'))}", flush=True)
        print(f"[output] val={validation_prediction.shape} test={test_prediction.shape} elapsed_seconds={time.time() - started:.1f}", flush=True)
        return
    panel = build_panel(ctx, debug)
    print(f"[runtime] phase=replay elapsed_seconds={time.time() - started:.1f}", flush=True)
    accepted, gate, fold_experts, champion = run_gate(ctx, panel, debug)
    print(f"[runtime] phase=forward_gate accepted={accepted} elapsed_seconds={time.time() - started:.1f}", flush=True)
    validation_prediction, test_prediction, diagnostics = final_predictions(ctx, panel, accepted, gate, fold_experts, champion, debug)
    validation_prediction = np.clip(np.asarray(validation_prediction, dtype=np.float32), 1e-6, 1 - 1e-6)
    test_prediction = np.clip(np.asarray(test_prediction, dtype=np.float32), 1e-6, 1 - 1e-6)
    if validation_prediction.shape != (247398,) or test_prediction.shape != (255360,):
        raise RuntimeError(f"prediction shape mismatch: {validation_prediction.shape} {test_prediction.shape}")
    if not np.isfinite(validation_prediction).all() or not np.isfinite(test_prediction).all():
        raise RuntimeError("predictions contain non-finite values")
    output = run_data_dir()
    np.save(output / "val_predictions.npy", validation_prediction)
    np.save(output / "test_predictions.npy", test_prediction)
    diagnostics["fingerprints"] = fingerprints
    diagnostics["mode"] = "debug" if debug else "full"
    diagnostics["elapsed_seconds"] = time.time() - started
    (output / "metrics.json").write_text(json.dumps(diagnostics, indent=2))
    if diagnostics["fallback"] is None and not debug:
        frozen.write_text(json.dumps(diagnostics, indent=2))
    register_artifact()
    print(f"[diagnostics] {json.dumps(diagnostics, separators=(',', ':'))}", flush=True)
    print(f"[output] val={validation_prediction.shape} test={test_prediction.shape} elapsed_seconds={time.time() - started:.1f}", flush=True)


if __name__ == "__main__":
    main()
