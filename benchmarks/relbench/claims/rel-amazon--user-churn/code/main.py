from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from kapso_datasets.common import save_predictions
from transition_specialist import json_value, run_transition_specialist


def main() -> None:
    warnings.filterwarnings("ignore")
    np.random.seed(1337)
    started = time.time()
    debug = "--debug" in sys.argv
    print(
        f"[transition-specialist] configuration debug={debug} threads={os.environ.get('OMP_NUM_THREADS')} "
        "hash_width=1048576 sketches=16x2 half_lives=180/730 trees=500 router=0.25",
        flush=True,
    )
    val, test, diagnostics = run_transition_specialist(debug, started)
    val = np.clip(np.nan_to_num(val, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
    test = np.clip(np.nan_to_num(test, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
    save_predictions(val, test)
    diagnostics["elapsed_seconds"] = time.time() - started
    diagnostics["val_shape"] = list(val.shape)
    diagnostics["test_shape"] = list(test.shape)
    diagnostics["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    output = Path("output_data_generic_exp_9")
    output.mkdir(parents=True, exist_ok=True)
    run_dir = Path(os.environ.get("KAPSO_RUN_DATA_DIR", output))
    payload = json.dumps(diagnostics, indent=2, sort_keys=True, default=json_value)
    (run_dir / "metrics.json").write_text(payload)
    (output / ("debug_metrics.json" if debug else "metrics.json")).write_text(payload)
    print(f"[transition-specialist] phase=complete elapsed={time.time() - started:.1f}s val={val.shape} test={test.shape}", flush=True)


if __name__ == "__main__":
    main()
