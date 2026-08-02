import hashlib
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from price_panel import PricePanelPipeline


# Orchestration

def elapsed(start, phase):
    value = time.time() - start
    print(f"[timing] {phase}: {value:.1f}s", flush=True)


def main():
    warnings.filterwarnings("ignore")
    start = time.time()
    debug = "--debug" in sys.argv
    run_dir = Path(os.environ.get("KAPSO_RUN_DATA_DIR", "output_data_generic_exp_2"))
    run_dir.mkdir(parents=True, exist_ok=True)
    pipeline = PricePanelPipeline(debug=debug)
    elapsed(start, "data initialization")

    val_rows = pipeline.load_inference_rows("val")
    test_rows = pipeline.load_inference_rows("test")
    val_origin = val_rows["t_dat"].min().normalize() - np.timedelta64(1, "D")
    test_origin = test_rows["t_dat"].min().normalize() - np.timedelta64(1, "D")

    val_predictions, diagnostics_a = pipeline.run_chain(
        chain_name="model_a",
        label_splits=("train",),
        inference_rows=val_rows,
        inference_origin=val_origin,
    )
    val_predictions = np.asarray(val_predictions, dtype=np.float64).copy()
    np.save(run_dir / "val_predictions.npy", val_predictions)
    val_hash = hashlib.sha256(val_predictions.tobytes()).hexdigest()[:16]
    print(f"[contract] Model A validation predictions frozen sha256={val_hash}", flush=True)
    elapsed(start, "Model A frozen")

    test_predictions, diagnostics_b = pipeline.run_chain(
        chain_name="model_b",
        label_splits=("train", "val"),
        inference_rows=test_rows,
        inference_origin=test_origin,
    )
    test_predictions = np.asarray(test_predictions, dtype=np.float64)
    if hashlib.sha256(val_predictions.tobytes()).hexdigest()[:16] != val_hash:
        raise RuntimeError("frozen validation predictions changed after Model B")
    np.save(run_dir / "val_predictions.npy", val_predictions)
    np.save(run_dir / "test_predictions.npy", test_predictions)
    pipeline.write_diagnostics(run_dir, diagnostics_a, diagnostics_b)
    elapsed(start, "predictions written")
    print(
        f"[contract] val{val_predictions.shape} test{test_predictions.shape} "
        f"debug={debug} panel_prices=task_labels_only",
        flush=True,
    )


if __name__ == "__main__":
    main()
