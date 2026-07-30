"""Finding 15 deep-dive: is the val->test inversion noise, or structural?

One-way local analysis. Never fed back into a running campaign.
"""
import glob
import json
import os

import numpy as np
from relbench.tasks import get_task
from scipy import stats
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(1337)

task = get_task("rel-event", "user-ignore", download=True)
tgt = task.target_col
val_y = task.get_table("val").df[tgt].to_numpy(dtype=float)
test_y = task.get_table("test", mask_input_cols=False).df[tgt].to_numpy(dtype=float)

runs = []
for d in sorted(glob.glob(f"{HERE}/peek3/runs/run_*")):
    vf, tf = f"{d}/val_predictions.npy", f"{d}/test_predictions.npy"
    if not (os.path.exists(vf) and os.path.exists(tf)):
        continue
    vp, tp = np.load(vf), np.load(tf)
    runs.append((os.path.basename(d), vp, tp,
                 roc_auc_score(val_y, vp) * 100, roc_auc_score(test_y, tp) * 100))

print(f"loaded {len(runs)} runs\n")

# --- 1. duplicates: identical prediction vectors are not independent samples
seen, uniq = {}, []
for name, vp, tp, v, t in runs:
    key = (round(v, 6), round(t, 6))
    if key in seen:
        seen[key].append(name)
        continue
    seen[key] = [name]
    uniq.append((name, v, t))
print("=== 1. DUPLICATE CANDIDATES (identical val+test) ===")
for key, names in seen.items():
    if len(names) > 1:
        print(f"  val={key[0]:.2f} test={key[1]:.2f}: {', '.join(names)}")
print(f"  {len(runs)} runs -> {len(uniq)} unique candidates\n")

uv = np.array([r[1] for r in uniq])
ut = np.array([r[2] for r in uniq])

# --- 2. correlation on unique candidates, with a permutation test
print("=== 2. VAL/TEST ASSOCIATION (unique candidates) ===")
pear, pp = stats.pearsonr(uv, ut)
spear, sp = stats.spearmanr(uv, ut)
print(f"  Pearson  r={pear:+.3f} (p={pp:.3f})")
print(f"  Spearman r={spear:+.3f} (p={sp:.3f})")
perm = np.array([stats.pearsonr(uv, rng.permutation(ut))[0] for _ in range(20000)])
print(f"  permutation: P(r <= observed | no association) = {(perm <= pear).mean():.4f}")
print(f"  -> val is {'ANTI-correlated' if pear < 0 else 'correlated'} with test; "
      f"{'not distinguishable from chance' if pp > 0.05 else 'significant'}\n")

# --- 3. is the val-max family's val edge within noise?
print("=== 3. IS THE VAL EDGE NOISE? (bootstrap AUROC standard errors) ===")


def boot_se(y, p, reps=400):
    n = len(y)
    out = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        out.append(roc_auc_score(y[idx], p[idx]))
    return np.std(out) * 100


by_name = {r[0]: r for r in runs}
champ, chall = by_name["run_0030"], by_name["run_0014"]
se_v30 = boot_se(val_y, champ[1])
se_v14 = boot_se(val_y, chall[1])
se_t30 = boot_se(test_y, champ[2])
se_t14 = boot_se(test_y, chall[2])
print(f"  run_0030 val {champ[3]:.2f} (SE {se_v30:.2f})  test {champ[4]:.2f} (SE {se_t30:.2f})")
print(f"  run_0014 val {chall[3]:.2f} (SE {se_v14:.2f})  test {chall[4]:.2f} (SE {se_t14:.2f})")
d_val = champ[3] - chall[3]
d_test = chall[4] - champ[4]
# paired bootstrap on the same resampled rows -> SE of the difference
dv, dt = [], []
for _ in range(400):
    i = rng.integers(0, len(val_y), len(val_y))
    j = rng.integers(0, len(test_y), len(test_y))
    if len(np.unique(val_y[i])) > 1:
        dv.append(roc_auc_score(val_y[i], champ[1][i]) - roc_auc_score(val_y[i], chall[1][i]))
    if len(np.unique(test_y[j])) > 1:
        dt.append(roc_auc_score(test_y[j], chall[2][j]) - roc_auc_score(test_y[j], champ[2][j]))
print(f"  val edge for 0030: {d_val:+.2f} pts (paired SE {np.std(dv) * 100:.2f}) "
      f"-> {abs(d_val) / (np.std(dv) * 100):.1f} sigma  {'REAL, not noise' if abs(d_val) / (np.std(dv) * 100) > 2 else 'within noise'}")
print(f"  test edge for 0014: {d_test:+.2f} pts (paired SE {np.std(dt) * 100:.2f}) "
      f"-> {abs(d_test) / (np.std(dt) * 100):.1f} sigma  {'REAL, not noise' if abs(d_test) / (np.std(dt) * 100) > 2 else 'within noise'}\n")

# --- 4. counterfactual selection rules
print("=== 4. WHAT DIFFERENT SELECTION RULES WOULD HAVE SHIPPED ===")
med, sd = np.median(uv), np.std(uv)
rules = {
    "argmax(val)  [current]": max(uniq, key=lambda r: r[1]),
    "median val candidate": sorted(uniq, key=lambda r: r[1])[len(uniq) // 2],
    "argmax(val) after dropping val-outliers >2sd": max(
        [r for r in uniq if r[1] <= med + 2 * sd] or uniq, key=lambda r: r[1]),
    "oracle argmax(test)  [unreachable]": max(uniq, key=lambda r: r[2]),
}
for label, (name, v, t) in rules.items():
    print(f"  {label:46s} -> {name}  val {v:.2f}  test {t:.2f}")
print(f"  mean test over all unique candidates: {ut.mean():.2f}")
print(f"  -> current rule is {ut.mean() - rules['argmax(val)  [current]'][2]:+.2f} vs simply picking at random\n")

# --- 5. does argmax(val) fail on the other peeked tasks too?
print("=== 5. CROSS-TASK: does argmax(val) pick the best test run? ===")
SPEC = {
    "rel-event--user-attendance": ("rel-event", "user-attendance", "mae"),
    "rel-f1--driver-position": ("rel-f1", "driver-position", "mae"),
    "rel-event--user-repeat": ("rel-event", "user-repeat", "auroc"),
}
for key, (ds, tn, metric) in SPEC.items():
    dirs = sorted(glob.glob(f"{HERE}/peek2/{key}/run_*"))
    if not dirs:
        continue
    tk = get_task(ds, tn, download=True)
    c = tk.target_col
    vy = tk.get_table("val").df[c].to_numpy(dtype=float)
    ty = tk.get_table("test", mask_input_cols=False).df[c].to_numpy(dtype=float)
    rows = []
    for d in dirs:
        vp, tp = np.load(f"{d}/val_predictions.npy"), np.load(f"{d}/test_predictions.npy")
        if vp.shape != vy.shape or tp.shape != ty.shape:
            continue
        if metric == "mae":
            rows.append((os.path.basename(d), float(np.abs(vp - vy).mean()), float(np.abs(tp - ty).mean())))
        else:
            rows.append((os.path.basename(d), roc_auc_score(vy, vp) * 100, roc_auc_score(ty, tp) * 100))
    if len(rows) < 2:
        continue
    better = (lambda a, b: a < b) if metric == "mae" else (lambda a, b: a > b)
    pick = sorted(rows, key=lambda r: r[1], reverse=(metric != "mae"))[0]
    best = sorted(rows, key=lambda r: r[2], reverse=(metric != "mae"))[0]
    vs = np.array([r[1] for r in rows])
    ts = np.array([r[2] for r in rows])
    r_, _ = stats.pearsonr(vs, ts)
    verdict = "OPTIMAL" if pick[0] == best[0] else f"suboptimal (left {abs(best[2] - pick[2]):.4f})"
    print(f"  {ds}/{tn:18s} n={len(rows):2d}  corr(val,test)={r_:+.3f}  "
          f"argmax(val)={pick[0]} test={pick[2]:.4f}  best={best[0]} test={best[2]:.4f}  -> {verdict}")
