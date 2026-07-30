"""One-way local peek: user-ignore val/test score flow across the live run sequence.

For OUR eyes only — never fed back into the running campaign.
"""
import glob
import os

import numpy as np
from relbench.tasks import get_task
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = HERE + "/peek3/runs"

task = get_task("rel-event", "user-ignore", download=True)
target = task.target_col
val_y = task.get_table("val").df[target].to_numpy(dtype=float)
test_y = task.get_table("test", mask_input_cols=False).df[target].to_numpy(dtype=float)
print(f"user-ignore  val n={len(val_y)} pos={val_y.mean():.3f} | test n={len(test_y)} pos={test_y.mean():.3f}")
print(f"bar: 91.2 (PluRel-ft)   prior-campaign best test: 80.59\n")
print(f"{'run':10s} {'val':>7s} {'test':>7s} {'gap':>7s}  {'val-max so far':>14s} {'its test':>9s}")

rows = []
best_v = None
for d in sorted(glob.glob(f"{RUNS}/run_*")):
    run = os.path.basename(d)
    vf, tf = f"{d}/val_predictions.npy", f"{d}/test_predictions.npy"
    if not (os.path.exists(vf) and os.path.exists(tf)):
        print(f"{run:10s} {'—':>7s} {'—':>7s}   (no predictions — crashed or in flight)")
        continue
    vp, tp = np.load(vf), np.load(tf)
    if vp.shape != val_y.shape or tp.shape != test_y.shape:
        print(f"{run:10s} shape mismatch val{vp.shape} test{tp.shape}")
        continue
    v, t = roc_auc_score(val_y, vp) * 100, roc_auc_score(test_y, tp) * 100
    rows.append((run, v, t))
    if best_v is None or v > best_v[1]:
        best_v = (run, v, t)
    print(f"{run:10s} {v:7.2f} {t:7.2f} {t - v:+7.2f}  {best_v[0]:>14s} {best_v[2]:9.2f}")

if rows:
    sel = max(rows, key=lambda r: r[1])
    oracle = max(rows, key=lambda r: r[2])
    print(f"\nargmax(val) would ship {sel[0]}: val {sel[1]:.2f} -> test {sel[2]:.2f}")
    print(f"best achievable test    {oracle[0]}: val {oracle[1]:.2f} -> test {oracle[2]:.2f}")
    print(f"selection cost: {oracle[2] - sel[2]:.2f} AUROC points left on the table")
    vs = np.array([r[1] for r in rows])
    ts = np.array([r[2] for r in rows])
    print(f"corr(val, test) = {np.corrcoef(vs, ts)[0, 1]:+.3f}   "
          f"val range [{vs.min():.2f}, {vs.max():.2f}]  test range [{ts.min():.2f}, {ts.max():.2f}]")
