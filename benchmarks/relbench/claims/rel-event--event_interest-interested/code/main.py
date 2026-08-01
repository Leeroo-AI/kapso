import json
import os
import time
from pathlib import Path

import numpy as np

from rel_event_model import run_pipeline


def main() -> None:
    started = time.time()
    debug = "--debug" in os.sys.argv
    output_dir = Path(os.environ["KAPSO_RUN_DATA_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_pipeline(debug=debug, started=started)
    val_predictions = np.clip(np.asarray(result["val_predictions"], dtype=np.float64), 1e-6, 1.0 - 1e-6)
    test_predictions = np.clip(np.asarray(result["test_predictions"], dtype=np.float64), 1e-6, 1.0 - 1e-6)
    if val_predictions.shape != (536,) or test_predictions.shape != (420,):
        raise RuntimeError(f"prediction shapes are {val_predictions.shape} and {test_predictions.shape}")
    if not np.isfinite(val_predictions).all() or not np.isfinite(test_predictions).all():
        raise RuntimeError("predictions contain non-finite values")
    np.save(output_dir / "val_predictions.npy", val_predictions)
    np.save(output_dir / "test_predictions.npy", test_predictions)
    (output_dir / "metrics.json").write_text(json.dumps(result["diagnostics"], indent=2))
    print(f"[prediction checks] val={val_predictions.shape} test={test_predictions.shape} elapsed={time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
