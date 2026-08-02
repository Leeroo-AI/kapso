import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from kapso_datasets.common import load_task, run_data_dir
from ranking_solution import AffinityPipeline


def elapsed(start):
    return f"{time.time() - start:.1f}s"


def main():
    start = time.time()
    debug = "--debug" in sys.argv
    np.random.seed(1337)
    print(f"[ranking] mode={'debug' if debug else 'full'}")
    context = load_task()
    print(f"[ranking] loaded task in {elapsed(start)}")
    pipeline = AffinityPipeline(context, debug=debug, seed=1337)
    result = pipeline.run()
    destination = run_data_dir()
    np.save(destination / "val_predictions.npy", result["val_predictions"])
    np.save(destination / "test_predictions.npy", result["test_predictions"])
    diagnostics = {k: v for k, v in result.items() if not k.endswith("_predictions")}
    (destination / "metrics.json").write_text(json.dumps(diagnostics, indent=2))
    output_dir = Path("output_data_generic_exp_1")
    output_dir.mkdir(exist_ok=True)
    (output_dir / "last_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    print(f"[ranking] wrote val{result['val_predictions'].shape} test{result['test_predictions'].shape} in {elapsed(start)}")


if __name__ == "__main__":
    main()
