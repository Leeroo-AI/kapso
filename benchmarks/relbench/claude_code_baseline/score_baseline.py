"""One-way test scoring for a harvested Claude Code baseline run.

Usage: score_baseline.py <dataset>/<task> <run_dir>

Loads the task from the canonical (pristine) relbench cache, validates the
prediction contract, scores val and test with relbench's own task.evaluate,
and writes <run_dir>/final_report.json in the campaign's report shape.
"""
import json
import sys
from pathlib import Path

import numpy as np

from benchmarks.relbench.task_specs import resolve_spec
from relbench.tasks import get_task

tname, run_dir = sys.argv[1], Path(sys.argv[2])
ds, name = tname.split("/")
task = get_task(ds, name, download=True)
spec = resolve_spec(task, ds, name)

out = run_dir / "kapso_output"
report = {"dataset": ds, "task": name, "primary_metric": spec.primary_metric,
          "run": run_dir.name, "val_metrics": {}, "test_metrics": {}}
for split in ("val", "test"):
    p = out / f"{split}_predictions.npy"
    if not p.exists():
        report[f"{split}_error"] = "missing predictions"
        continue
    pred = np.load(p, allow_pickle=True)
    table = task.get_table(split, mask_input_cols=False)
    expected = spec.expected_pred_shape(len(table.df))
    if tuple(pred.shape) != tuple(expected):
        report[f"{split}_error"] = f"shape {pred.shape} != expected {expected}"
        continue
    if spec.family.endswith("classification") and not spec.is_multiclass and spec.target_col:
        y = table.df[spec.target_col]
        if y.dtype == object:
            table.df[spec.target_col] = (y == "t").astype(int)
    report[f"{split}_metrics"] = {k: float(v) for k, v in task.evaluate(pred, table).items()}

(run_dir / "final_report.json").write_text(json.dumps(report, indent=1))
pm = spec.primary_metric
val = report["val_metrics"].get(pm)
test = report["test_metrics"].get(pm)
print(f"{tname}: val {pm}={val if val is None else round(val, 4)} "
      f"test {pm}={test if test is None else round(test, 4)} "
      f"{'(' + report.get('test_error', report.get('val_error', '')) + ')' if 'test_error' in report or 'val_error' in report else ''}")
