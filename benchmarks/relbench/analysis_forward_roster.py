"""Forward-roster coverage test for rel-trial/site-success (offline, one-way).

Questions:
 1. LINEAGE - is the per-trial success definition identical to study-outcome's?
 2. COVERAGE - what fraction of the trials that DEFINE the label (reporting a
    primary p-value analysis inside the label window) are already visible /
    registered at the cutoff, i.e. their protocol is readable at prediction time?
 3. CEILING - MAE/NMAE of a covered-roster predictor: oracle per-trial labels,
    and a simulated per-trial classifier at study-outcome-grade AUC (0.82),
    each with historical-rate fallback for the uncovered remainder - versus
    the historical-rate baseline the past campaigns plateaued at.
"""
import inspect
from statistics import NormalDist

import duckdb
import numpy as np
import pandas as pd

import relbench.tasks.trial as trial_mod
from relbench.datasets import get_dataset
from relbench.tasks import get_task

# 1. lineage: print study-outcome's success-definition lines
src = inspect.getsource(trial_mod.StudyOutcomeTask)
print("=== StudyOutcomeTask success-definition lines ===")
for line in src.splitlines():
    if any(k in line for k in ("p_value", "successful", "Primary")):
        print("   ", line.strip())

db = get_dataset("rel-trial", download=False).get_db()
task = get_task("rel-trial", "site-success", download=False)
train = task.get_table("train").df
val = task.get_table("val").df
std_train = train["success_rate"].std()
print(f"\ntrain rows={len(train)}  val rows={len(val)}  std(train)={std_train:.5f}")

outcome_analyses = db.table_dict["outcome_analyses"].df
outcomes = db.table_dict["outcomes"].df
facilities_studies = db.table_dict["facilities_studies"].df
studies = db.table_dict["studies"].df
scol = db.table_dict["studies"].time_col
print("studies.time_col =", scol)

ti = duckdb.sql("""
    SELECT oa.nct_id,
           MIN(CASE WHEN oa.p_value < 0.05 THEN 1 ELSE 0 END) AS is_successful,
           oa.date
    FROM outcome_analyses oa
    LEFT JOIN outcomes o ON oa.outcome_id = o.id
    WHERE (oa.p_value_modifier IS NULL OR oa.p_value_modifier != '>')
      AND oa.p_value >= 0 AND oa.p_value <= 1
      AND o.outcome_type = 'Primary'
    GROUP BY oa.nct_id, oa.date
""").df()
ti["date"] = pd.to_datetime(ti["date"])
vis = studies[["nct_id", scol]].rename(columns={scol: "svis"})
vis["svis"] = pd.to_datetime(vis["svis"])
fs = facilities_studies[["facility_id", "nct_id"]]

AUC = 0.82
DP = float(np.sqrt(2)) * NormalDist().inv_cdf(AUC)


def analyze(t0, label_df, tag, rng):
    t1 = t0 + pd.Timedelta(days=365)
    win = ti[(ti["date"] > t0) & (ti["date"] <= t1)].merge(vis, on="nct_id", how="left")
    win["covered"] = (win["svis"] <= t0).fillna(False)
    winf = win.merge(fs, on="nct_id", how="inner")
    lab = label_df[["facility_id", "success_rate"]].drop_duplicates("facility_id")
    winf = winf[winf["facility_id"].isin(set(lab["facility_id"]))].copy()

    pi = winf["is_successful"].mean()
    s = rng.normal(0, 1, len(winf)) + DP * winf["is_successful"].to_numpy()
    p = 1.0 / (1.0 + ((1 - pi) / pi) * np.exp(-DP * s + DP * DP / 2))
    winf["p_sim"] = np.where(winf["covered"], p, np.nan)
    winf["y_cov"] = winf["is_successful"].where(winf["covered"])

    g = winf.groupby("facility_id").agg(
        n=("is_successful", "size"),
        y_all=("is_successful", "mean"),
        ncov=("covered", "sum"),
        y_covmean=("y_cov", "mean"),
        p_covmean=("p_sim", "mean"),
    ).reset_index()
    m = lab.merge(g, on="facility_id", how="left")
    recon = float((m["y_all"] - m["success_rate"]).abs().max())

    hist = ti[ti["date"] <= t0].merge(fs, on="nct_id", how="inner")
    h = hist.groupby("facility_id")["is_successful"].agg(hrate="mean", hn="size").reset_index()
    gmean = hist["is_successful"].mean()
    m = m.merge(h, on="facility_id", how="left")
    m["hn"] = m["hn"].fillna(0)
    k = 5.0
    m["fb"] = (m["hn"] * m["hrate"].fillna(0) + k * gmean) / (m["hn"] + k)

    n = m["n"].fillna(0).astype(float)
    ncov = m["ncov"].fillna(0).astype(float)
    nsafe = n.replace(0, 1)

    def mix(colmean):
        covpart = colmean.fillna(0) * ncov
        rest = (n - ncov) * m["fb"]
        return pd.Series(np.where(n > 0, (covpart + rest) / nsafe, m["fb"]), index=m.index)

    pred_hist = m["fb"]
    pred_oracle = mix(m["y_covmean"])
    pred_real = mix(m["p_covmean"])
    pred_real_q = pd.Series(np.where(n > 0, np.round(pred_real * n) / nsafe, pred_real), index=m.index)

    y = m["success_rate"]
    mae = lambda pr: float(np.abs(pr - y).mean())
    cold = m["hn"] == 0
    out = {
        "tag": tag, "t0": str(pd.Timestamp(t0).date()), "facilities": len(m),
        "window_trials": int(win["nct_id"].nunique()),
        "cov_trials": float(win["covered"].mean()),
        "cov_rows": float(winf["covered"].mean()),
        "med_roster": float(n.median()), "share_n1": float((n == 1).mean()),
        "cold": int(cold.sum()), "recon": recon,
        "MAE_global": float(np.abs(gmean - y).mean()),
        "MAE_hist": mae(pred_hist),
        "MAE_oracle": mae(pred_oracle),
        "MAE_sim82": mae(pred_real),
        "MAE_sim82_q": mae(pred_real_q),
    }
    if cold.any():
        out["MAE_hist_cold"] = float(np.abs(pred_hist[cold] - y[cold]).mean())
        out["MAE_sim82_cold"] = float(np.abs(pred_real[cold] - y[cold]).mean())
        out["share_cold"] = float(cold.mean())
    return out


rng = np.random.default_rng(7)
t0_val = pd.to_datetime(val["timestamp"]).max()
t_last = pd.to_datetime(train["timestamp"]).max()
results = [
    analyze(t0_val, val, "VAL", rng),
    analyze(t_last, train[pd.to_datetime(train["timestamp"]) == t_last], "TRAIN-LAST", rng),
]

print("\n=== forward-roster test (NMAE = MAE / %.5f) ===" % std_train)
for r in results:
    print(f"\n[{r['tag']}] t0={r['t0']}  facilities={r['facilities']}  window-trials={r['window_trials']}")
    print(f"  coverage: trials={r['cov_trials']:.1%}  facility-rows={r['cov_rows']:.1%}   "
          f"median-roster={r['med_roster']:.0f}  share(n=1)={r['share_n1']:.1%}  "
          f"cold-facilities={r['cold']} ({r.get('share_cold', 0):.1%})")
    print(f"  label-reconstruction max err: {r['recon']:.5f}")
    for kk in ("MAE_global", "MAE_hist", "MAE_oracle", "MAE_sim82", "MAE_sim82_q"):
        print(f"  {kk:12s} = {r[kk]:.4f}   NMAE = {r[kk] / std_train:.4f}")
    if "MAE_hist_cold" in r:
        print(f"  cold slice:  hist={r['MAE_hist_cold']:.4f}  sim82={r['MAE_sim82_cold']:.4f}")

print("\nreference: banked NMAE 0.778 | field-best RT 0.5519 | campaign val-NMAE 0.769")
