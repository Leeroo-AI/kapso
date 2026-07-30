import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from avito_pipeline import AvitoPipeline
from tabpfn_specialist import TabPFNSpecialist


def main() -> None:
    started = time.time()
    debug = "--debug" in sys.argv
    pipeline = AvitoPipeline(debug)
    prepared = pipeline.prepare()
    print(f"[phase] prepared elapsed={time.time() - started:.1f}s")
    specialist = TabPFNSpecialist(pipeline)
    selection = specialist.forward_select(prepared)
    print(f"[phase] forward_cv elapsed={time.time() - started:.1f}s")
    val_predictions = specialist.fit_model_a(prepared, selection)
    output_dir = Path(os.environ.get("KAPSO_RUN_DATA_DIR", "./kapso_output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "val_predictions.npy", val_predictions)
    print(f"[phase] model_a_saved elapsed={time.time() - started:.1f}s")
    test_predictions = specialist.fit_model_b(prepared, selection)
    np.save(output_dir / "test_predictions.npy", test_predictions)
    metrics = specialist.diagnostics(prepared, selection)
    metrics["elapsed_seconds"] = time.time() - started
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    specialist.record_campaign(metrics)
    print(
        f"[complete] val={val_predictions.shape} test={test_predictions.shape} "
        f"debug={debug} elapsed={time.time() - started:.1f}s"
    )


if __name__ == "__main__":
    main()
