import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from graph_moe import run_pipeline


def main() -> None:
    warnings.filterwarnings("ignore")
    started = time.time()
    debug = "--debug" in sys.argv
    result = run_pipeline(debug)
    out = Path(os.environ["KAPSO_RUN_DATA_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "val_predictions.npy", result.val_predictions.astype(np.float64))
    np.save(out / "test_predictions.npy", result.test_predictions.astype(np.float64))
    with (out / "metrics.json").open("w") as handle:
        json.dump(result.diagnostics, handle, sort_keys=True, indent=2)
    local = Path("output_data_generic_exp_1")
    local.mkdir(parents=True, exist_ok=True)
    with (local / ("debug_metrics.json" if debug else "full_metrics.json")).open("w") as handle:
        json.dump(result.diagnostics, handle, sort_keys=True, indent=2)
    elapsed = time.time() - started
    print(f"[complete] debug={debug} val={result.val_predictions.shape} test={result.test_predictions.shape} elapsed_s={elapsed:.2f}")


if __name__ == "__main__":
    main()
